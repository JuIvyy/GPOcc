import torch
import numpy as np
from copy import deepcopy
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG
import sys
from depth_anything_v2.dpt import DepthAnythingV2
import torch.nn as nn
from PIL import Image
import cv2
import torch.nn.functional as F
import os
import numpy as np
import copy
import matplotlib.pyplot as plt
from einops import einsum, repeat, rearrange
from torch_cluster import radius
from torch_scatter import scatter_add, scatter_max, scatter_min

from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix

try:
    from torch_geometric.nn import radius_graph
except ImportError as thgeo_import_error:
    radius_graph = None

from ...encoder.gaussianformer.utils import \
    cartesian, safe_sigmoid, GaussianPrediction, get_rotation_matrix, safe_get_quaternion
from ...encoder.gaussianformer.gaussian_encoder_layer import SparseGaussian3DEncoder

try:
    from ...encoder.gaussianformer.ops import DeformableAggregationFunction as DAF
except ImportError:
    DAF = None

from gpocc.model.vggt.heads.head_act import inverse_log_transform
from gpocc.model.head.gaussian_occ_head.gsplat_rasterization import rasterize_gaussians

from gpocc.model.vggt.layers import PatchEmbed
from .merge_utils import DiffGaussianUpdaterSparsePerA


try:
    from torch_cluster import radius
except ImportError as torch_cluster_import_e:
    radius = None

try:
    import torch_scatter
except ImportError as torch_scatter_import_e:
    torch_scatter = None

try:
    import cupy as cp
    import cudf
    import cugraph
except ImportError as cu_pkg_import_error:
    cugraph = cudf = cp = None


def inverse_sigmoid(x):
    return torch.log(x/((1-x)+1e-10))


@MODELS.register_module()
class GaussianSegmentor(BaseModule):

    def __init__(
        self,
        flag_depthbranch=False,
        flag_depthanything_as_gt=False,
        depthbranch=None,
        backbone=None,
        neck=None,
        lifter=None,
        encoder=None,
        future_decoder=None,
        head=None, 
        init_cfg=None,
        **kwargs,
    ):
        super().__init__(init_cfg)
        self.flag_depthbranch = flag_depthbranch
        self.flag_depthanything_as_gt = flag_depthanything_as_gt
        if flag_depthbranch:
            if flag_depthanything_as_gt:
                model_configs = {
                    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
                }
                self.depthanything = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth':20})
                checkpoint = torch.load(f"{os.getenv('HF_HOME', os.path.expanduser('~') + '/.cache/huggingface')}/hub/EmbodiedOcc/finetune_scannet_depthanythingv2.pth", map_location='cpu')['model']
                new_state_dict = {}
                for k, v in checkpoint.items():
                    if k.startswith('module.'):
                        new_key = k[len('module.'):] 
                    else:
                        new_key = k
                    new_state_dict[new_key] = v
                self.depthanything.load_state_dict(new_state_dict)

            basemodel_name = "tf_efficientnet_b7_ns"
            num_features = 2560
            print("Loading base model ()...".format(basemodel_name), end="")
            torch.hub._validate_not_a_forked_repo=lambda a,b,c: True
            basemodel = torch.hub.load(
                "rwightman/gen-efficientnet-pytorch", basemodel_name, pretrained=True
            )
            print("Done.")
            print("Removing last two layers (global_pool & classifier).")
            basemodel.global_pool = nn.Identity()
            basemodel.classifier = nn.Identity()

            self.backbone = basemodel

            self.neck = DecoderBN(
                out_feature=96,
                use_decoder=True,
                bottleneck_features=num_features,
                num_features=num_features,
            )
        else:
            pass
        if lifter is not None:
            self.lifter = MODELS.build(lifter)
        if encoder is not None:
            self.encoder = MODELS.build(encoder)
        if future_decoder is not None: 
            self.future_decoder = MODELS.build(future_decoder)
        if head is not None:
            self.head = MODELS.build(head)

    def extract_img_feat(self, imgs):
        B, N, C, H, W = imgs.size()
        imgs = imgs.reshape(B * N, C, H, W) # 1, 3, 480, 640
        
        feature_x = [imgs]
        feature_idx = 0
        this_x = feature_x[-1]
        for k, v in self.backbone._modules.items():
            if k == "blocks":
                for ki, vi in v._modules.items():
                    this_x = vi(this_x)
                    feature_idx += 1
                    if feature_idx in [4, 5, 6, 8, 11]:
                        feature_x.append(this_x)
            else:
                this_x = v(this_x)
                feature_idx += 1
                if feature_idx in [4, 5, 6, 8, 11]:
                    feature_x.append(this_x)
            
        img_feats_backbone = feature_x

        img_feats_out = self.neck(img_feats_backbone) # dict

        img_feats_reshaped = []
        for img_feat in img_feats_out.values():
            BN, C, H, W = img_feat.size()
            if W != 640:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))

        return img_feats_reshaped, img_feats_out['1_1'] # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]

    def obtain_bev(self, imgs, metas):
        B, f, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*f, N, C, H, W)

        mlvl_img_feats, feature_x_4 = self.extract_img_feat(imgs) # list of [1, 1, 96, 28, 36], [1, 1, 96, 14, 18], [1, 1, 96, 7, 9]

        if self.flag_depthbranch:
            if self.flag_depthanything_as_gt:
                self.depthanything.eval()
                image_ = metas[0]['img_depthbranch']
                depth_pred = self.depthanything.infer_image(image_, 480, 640, 480)
                depthnet_output = depth_pred
            else:  
                depthnet_output = None
        else:
            depthnet_output = None

        anchor, instance_feature, depth2occ, depthnet_output_loss, predtoreturn = self.lifter(self.flag_depthbranch, self.flag_depthanything_as_gt, depthnet_output, mlvl_img_feats, metas)    # b, g, c 

        anchor, feats = self.encoder(anchor, instance_feature, mlvl_img_feats, metas) # b, g, c
        return anchor, depth2occ, depthnet_output_loss, predtoreturn, feats

    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        **kwargs,
    ):

        B, f, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'
        if grad_frames is not None:
            assert grad_frames < f
            imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index = self.frame_split(grad_frames, imgs, metas)
            bev_grad = self.obtain_bev(imgs_grad, metas_grad)
            with torch.no_grad():
                bev_no_grad = self.obtain_bev(imgs_no_grad, metas_no_grad)
            bev = torch.cat([bev_grad, bev_no_grad], dim=0)[inv_index]
            feats = None
        else:
            bev, depth2occ, depthnet_output_loss, predtoreturn, feats = self.obtain_bev(imgs, metas)

        feats = feats[-1] if isinstance(feats, list) else feats

        BF, G, C = bev.shape # bev is actually anchors [1, 21600, 24]
        bev = bev.reshape(B, f, G, C)
        if hasattr(self, 'future_decoder'):
            output_dict = self.future_decoder(bev, metas)
            bev_predict = output_dict.pop('bev')
        else:
            bev_predict = bev
            output_dict = dict()

        output_dict = self.head(
            bev_feat=bev_predict,  # [1, 1, 21600, 24]
            points=points,
            label=label,
            output_dict=output_dict, 
            metas=metas,
            inst_feats=feats,
            test_mode=test_mode)

        return output_dict, depth2occ, predtoreturn

    def frame_split(self, grad_frames, imgs, metas):
        f = imgs.shape[1]
        index = np.random.permutation(f)
        inv_index = np.argsort(index)
        imgs_grad = imgs[:, index[:grad_frames]]
        imgs_no_grad = imgs[:, index[grad_frames:]]
        metas_grad = deepcopy(metas)
        metas_no_grad = deepcopy(metas)
        for meta, meta_grad, meta_no_grad in zip(metas, metas_grad, metas_no_grad):
            lidar2img = np.asarray(meta['lidar2img'])
            meta_grad['lidar2img'] = lidar2img[index[:grad_frames]]
            meta_no_grad['lidar2img'] = lidar2img[index[grad_frames:]]
            img_aug_matrix = meta['img_aug_matrix']
            meta_grad['img_aug_matrix'] = img_aug_matrix[index[:grad_frames]]
            meta_no_grad['img_aug_matrix'] = img_aug_matrix[index[grad_frames:]]

        return imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index
    
    def forward_autoreg(self,
                        imgs=None,
                        metas=None,
                        points=None,
                        label=None,
                        test_mode=True,
                        **kwargs,
        ):
        B, f, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'

        bev = self.obtain_bev(imgs, metas)
        BF, G, C = bev.shape # bev is actually anchors
        bev = bev.reshape(B, f, G, C)

        output_dict = self.future_decoder.forward_autoreg(bev, metas)
        bev_predict = output_dict.pop('bev')
        output_dict = self.head(
            bev_feat=bev_predict, 
            points=points, 
            label=label, 
            output_dict=output_dict, 
            metas=metas,
            test_mode=test_mode)

        return output_dict


@MODELS.register_module()
class VGGTGaussianSegmentor(GaussianSegmentor):

    def __init__(
        self,
        *args,
        pretrained_path=None,
        text_prompts=None,
        frozen_backbone=True,
        freeze_blocks=-1,
        semantic_dim=13,
        scale_range=[0.01, 0.08],
        include_opa=True,
        cuda_kwargs=dict(
            scale_multiplier=3,
            H=200, W=200, D=16,
            pc_min=[-40.0, -40.0, -1.0],
            grid_size=0.4),
        num_bins=10,
        with_unc=False,
        extra_sparse_gaussian=None,
        opacities_threshold=0.,
        densities_threshold=None,
        deform_offset=False,
        use_depthanything=False,
        random_init=False, # only for ablation

        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        def _module_require_grad(m, flag=False):
            for n, p in m.named_parameters():
                p.requires_grad = flag

        self.use_depthanything = use_depthanything
        if use_depthanything:
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            depthanything = DepthAnythingV2(**{**model_configs['vitb'], 'max_depth':20})
            checkpoint = torch.load(f"{os.getenv('HF_HOME', os.path.expanduser('~') + '/.cache/huggingface')}/hub/EmbodiedOcc/finetune_scannet_depthanythingv2.pth", map_location='cpu')['model']
            new_state_dict = {}
            for k, v in checkpoint.items():
                if k.startswith('module.'):
                    new_key = k[len('module.'):]
                else:
                    new_key = k
                new_state_dict[new_key] = v
            depthanything.load_state_dict(new_state_dict)
            self.backbone = depthanything

            _module_require_grad(self.backbone, not frozen_backbone)

            self.gs_head = copy.deepcopy(self.backbone.depth_head)
            for n, p in self.gs_head.named_parameters():
                p.requires_grad = True
            self.backbone.depth_head = None

        else:
            from gpocc.model.vggt.models.vggt import VGGT

            pretrained_path = f"{os.getenv('HF_HOME', os.path.expanduser('~') + '/.cache/huggingface')}/hub/VGGT-1B"

            model = VGGT.from_pretrained(pretrained_path)
            self.backbone = model

            _module_require_grad(self.backbone, False)

            if not frozen_backbone:
                _module_require_grad(self.backbone.aggregator, True)

            self.freeze_blocks = freeze_blocks
            if freeze_blocks > 0:
                assert not frozen_backbone
                for idx in range(freeze_blocks):
                    _module_require_grad(
                        self.backbone.aggregator.frame_blocks[idx], True)
                    _module_require_grad(
                        self.backbone.aggregator.global_blocks[idx], True)

            self.gs_head = copy.deepcopy(self.backbone.depth_head)
            for n, p in self.gs_head.named_parameters():
                p.requires_grad = True

            self.backbone.depth_head = None
            self.backbone.camera_head = None
            self.backbone.point_head = None
            self.backbone.track_head = None

        self.frozen_backbone = frozen_backbone

        self.with_unc = with_unc

        self.scale_range = scale_range
        self.include_opa = include_opa
        self.semantic_dim = semantic_dim
        self.semantic_start = 10 + int(include_opa)

        self.cuda_kwargs = cuda_kwargs

        from gpocc.model.head.gaussian_occ_head.ops.localagg_prob_gf2.local_aggregate_prob_gf2 import LocalAggregator
        self.aggregator = LocalAggregator(**cuda_kwargs)

        _dim_ = 256
        self.gs_pred_layer = nn.Sequential(
            nn.Linear(_dim_, _dim_),
            nn.ReLU(),
            nn.Linear(_dim_,
                      semantic_dim + 3 + 4 + 1 + (1 if with_unc else 0))
        )
        self.anchor_feat_layer = nn.Sequential(
            nn.Conv2d(64 if self.use_depthanything else 128, _dim_, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(_dim_, _dim_, kernel_size=2, stride=2),
        )
        self.anchor_depth_pred = nn.Sequential(
            nn.Conv2d(_dim_, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )

        self.num_bins = num_bins
        if num_bins > 0:
            self.bins_emb = nn.Embedding(num_bins, _dim_)
            self.depth_scale_layer = nn.Conv2d(_dim_, 1, kernel_size=1)

        self.opacities_threshold = opacities_threshold
        self.densities_threshold = densities_threshold

        if densities_threshold is not None:
            assert opacities_threshold is None
        if opacities_threshold is not None:
            assert densities_threshold is None

        self.extra_sparse_gaussian = extra_sparse_gaussian
        if extra_sparse_gaussian:
            self.sparse_gs_pred_layer = nn.Sequential(
                nn.Linear(_dim_, _dim_),
                nn.ReLU(),
                nn.Linear(_dim_, _dim_),
                nn.ReLU(),
                nn.Linear(_dim_, _dim_),
                nn.ReLU(),
                nn.Linear(_dim_,
                        semantic_dim + 3 + 4 + (1 if with_unc else 0))
            )
            self.sparse_anchor_feat_layer = nn.Sequential(
                nn.Conv2d(128, _dim_, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Conv2d(_dim_, _dim_, kernel_size=2, stride=2),
            )
            self.sparse_bins_emb = nn.Embedding(num_bins, _dim_)

        if deform_offset:
            self.deform_offset_layer = nn.Sequential(
                nn.Linear(_dim_, _dim_),
                nn.ReLU(),
                nn.Linear(_dim_, 3)
            )
            nn.init.zeros_(self.deform_offset_layer[-1].weight)
            nn.init.zeros_(self.deform_offset_layer[-1].bias)
        self.deform_offset = deform_offset
        self.random_init = random_init

    def init_weights(self):
        if self.random_init:
            print("[init_weights] Random init all Conv/Linear layers (no pretrain).")
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                    nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

                elif isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward_backbone(self, imgs, extra_feat=None):
        predictions = dict()

        if self.use_depthanything:
            output = self.backbone.custom_forward(imgs.flatten(0, 1))
            return output
        else:
            aggregated_tokens_list, patch_start_idx = self.backbone.aggregator(
                images=imgs,
                freeze_blocks=None if self.frozen_backbone else self.freeze_blocks,
                extra_feat=extra_feat)
            predictions['aggregated_tokens_list'] = aggregated_tokens_list
            predictions['patch_start_idx'] = patch_start_idx

            return predictions

    def prepare_gaussian_args(self, gaussians, metas):

        means = gaussians.means # b, g, 3
        b_, g_, _ = means.shape
        means_cam = F.pad(means, (0, 1), value=1)
        cam2world = torch.stack([meta['cam2world'] for meta in metas]).float().cuda()
        means_world_ = einsum(cam2world, means_cam, 'b n k, b j k -> b j n')
        means_world = means_world_[..., :3]
        means = means_world
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1

        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)

        bs, g, _ = means.shape

        S = torch.zeros(bs, g, 3, 3, dtype=means.dtype, device=means.device)
        S[..., 0, 0] = scales[..., 0]
        S[..., 1, 1] = scales[..., 1]
        S[..., 2, 2] = scales[..., 2]

        R = get_rotation_matrix(rotations) # b, g, 3, 3

        M = torch.matmul(S, R)
        Cov = torch.matmul(M.transpose(-1, -2), M)

        c2w_rot = torch.stack([meta['cam2world'][:3, :3] for meta in metas]).cuda()
        c2w_rot_T = rearrange(c2w_rot, 'b h w -> b w h')
        c2w_rot = c2w_rot.unsqueeze(1).repeat(1, g, 1, 1).to(torch.float32)
        c2w_rot_T = c2w_rot_T.unsqueeze(1).repeat(1, g, 1, 1).to(torch.float32)
        Cov = torch.matmul(c2w_rot, torch.matmul(Cov, c2w_rot_T))

        CovInv = Cov.double().inverse().to(Cov.dtype)
        return means, origi_opa, opacities, scales, CovInv

    def forward(
        self,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        grad_frames=None,
        test_mode=False,
        extra_feat=None,
        return_gaussian=False,
        **kwargs,
    ):

        assert imgs.shape[2] == 1, f'#view == 1, but got {imgs.shape[2]}'
        imgs = imgs.squeeze(2)
        imgs = rearrange(imgs, 'b f h w c -> b f c h w')

        nyu_pc_min = torch.stack([meta['vox_origin'] for meta in metas]).cuda()
        scene_size = torch.stack([meta['scene_size'] for meta in metas]).cuda()
        nyu_pc_max = nyu_pc_min + scene_size

        sampled_xyz = torch.stack([meta['occ_xyz'] if isinstance(meta['occ_xyz'], torch.Tensor) else torch.from_numpy(meta['occ_xyz']).cuda() for meta in metas])


        if self.frozen_backbone:
            with torch.no_grad():
                self.backbone.eval()
                predictions = self.forward_backbone(imgs, extra_feat=extra_feat)
        else:
            predictions = self.forward_backbone(imgs, extra_feat=extra_feat)

        if self.use_depthanything:
            ori_anchor_feat = self.gs_head.custom_forward(*predictions)
            ori_anchor_feat = ori_anchor_feat.unflatten(0, imgs.shape[:2])
        else:
            ori_anchor_feat = self.gs_head.custom_forward(
                predictions['aggregated_tokens_list'],
                images=imgs,
                patch_start_idx=predictions['patch_start_idx'],
            )

        anchor_feat = self.anchor_feat_layer(ori_anchor_feat.flatten(0, 1))
        anchor_depth = self.anchor_depth_pred(anchor_feat)

        max_depth = 8
        anchor_depth = anchor_depth.sigmoid() * max_depth  # (B, 1, H, W)

        device = anchor_depth.device
        dtype = anchor_depth.dtype
        B, _, H, W = anchor_depth.shape
        iH, iW = imgs.shape[-2:]

        u, v = torch.meshgrid(
            torch.arange(W, device=device),
            torch.arange(H, device=device),
            indexing='xy'
        )  # (H, W)
        uv = torch.stack([u, v, torch.ones_like(u)], dim=0).float()  # (3, H, W)
        uv = uv.unsqueeze(0).repeat(B, 1, 1, 1).to(dtype)  # (B, 3, H, W)

        intr = torch.stack([meta['cam_k'] for meta in metas]).to(dtype).to(device)  # (B, 3, 3)
        fx = intr[:, 0, 0].view(B, 1, 1)
        fy = intr[:, 1, 1].view(B, 1, 1)
        cx = intr[:, 0, 2].view(B, 1, 1)
        cy = intr[:, 1, 2].view(B, 1, 1)

        x = (uv[:, 0] / W * iW - cx) / fx
        y = (uv[:, 1] / H * iH - cy) / fy
        z = torch.ones_like(x)
        dirs = torch.stack([x, y, z], dim=1)
        dirs = torch.nn.functional.normalize(dirs, dim=1, eps=1e-6)  # unit direction vectors

        xyz = dirs * anchor_depth  # element-wise scale by depth

        if self.num_bins > 0:

            depth_scale = self.depth_scale_layer(anchor_feat).sigmoid()  # (B, num_bins, H, W)
            depth_offset = torch.linspace(
                0, 5, self.num_bins, device=device, dtype=dtype
            )[None, :, None, None] * depth_scale  # (B, num_bins, H, W)

            dirs = dirs.unsqueeze(1).expand(-1, self.num_bins, -1, -1, -1)  # (B, num_bins, 3, H, W)
            depth_offset = depth_offset.unsqueeze(2)  # (B, num_bins, 1, H, W)

            xyz_offset = xyz.unsqueeze(1) + dirs * depth_offset  # (B, num_bins, 3, H, W)

            anchor_feat_offset = anchor_feat.unsqueeze(1) + self.bins_emb.weight[None, :, :, None, None]

        else:
            xyz_offset = xyz[:, None]
            anchor_feat_offset = anchor_feat[:, None]

        xyz = rearrange(xyz_offset, 'b n c h w -> b (n h w) c')
        feat = rearrange(anchor_feat_offset, 'b n c h w -> b (n h w) c')
        gs_feat = self.gs_pred_layer(feat)

        if self.deform_offset:
            assert False, 'not good'
            deform_offset = self.deform_offset_layer(feat)
            xyz = xyz + deform_offset

        gs_scales = safe_sigmoid(gs_feat[..., :3])
        gs_scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * gs_scales
        rot = gs_feat[..., 3:7]
        opas = safe_sigmoid(gs_feat[..., 7:8])
        shs = torch.zeros(*gs_feat.shape[:-1], 0, device=device, dtype=dtype)
        semantics = gs_feat[..., 8 : 8 + self.semantic_dim]

        if self.with_unc:
            conf = gs_feat[..., 8 + self.semantic_dim: 8 + self.semantic_dim + 1]
        else:
            conf = None

        gaussian = GaussianPrediction(
            means=xyz,
            scales=gs_scales,
            rotations=rot,
            opacities=opas,
            semantics=semantics,
            feat=feat,
            conf=conf
        )

        if return_gaussian:
            return dict(gaussian=gaussian), None, None

        result_dict = {}
        sampled_xyz = sampled_xyz.to(device=device, dtype=torch.float32)

        sparse_gaussians = []
        sparse_occ_preds = []
        sparse_labels = []
        sparse_masks = []

        means, origi_opa, opacities, scales, CovInv = self.prepare_gaussian_args(gaussian, metas)

        semantics = []
        densities = []
        sem_labels = []
        fov_mask = []
        confs = []

        if self.with_unc:
            opacities = torch.cat([opacities, conf], dim=-1)

        for i in range(len(sampled_xyz)):
            epsilon = 1e-3
            mask = (
                means[i, ..., 0] > (nyu_pc_min[i, 0]+epsilon)
            ) & (
                means[i, ..., 0] < (nyu_pc_max[i, 0]-epsilon)
            ) & (
                means[i, ..., 1] > (nyu_pc_min[i, 1]+epsilon)
            ) & (
                means[i, ..., 1] < (nyu_pc_max[i, 1]-epsilon)
            ) & (
                means[i, ..., 2] > (nyu_pc_min[i, 2]+epsilon)
            ) & (
                means[i, ..., 2] < (nyu_pc_max[i, 2]-epsilon)
            )

            if self.training and mask.sum() < 100:
                print(f'skip due to only {mask.sum()} points inside range')
                continue

            if not self.training:
                mask = mask & (origi_opa[i].squeeze(-1) > self.opacities_threshold)

            if not self.training and mask.sum() == 0:
                semantic = torch.zeros(
                    *sampled_xyz[i:(i+1)].shape[:4],
                    opacities.shape[-1],
                    device=opacities.device,
                    dtype=opacities.dtype).flatten(0, -2)
                bin_logit = torch.zeros_like(semantic[..., 0])
                density = torch.zeros_like(semantic[..., 0])
            else:
                semantic, bin_logit, density = self.aggregator(
                    sampled_xyz[i:(i+1)].flatten(1, 3),
                    means[i][mask][None],
                    origi_opa[i][mask][None],
                    opacities[i][mask][None],
                    scales[i][mask][None],
                    CovInv[i][mask][None],
                    metas[i],
                    nyu_pc_min[i]) # n, c

            if self.with_unc:
                sem = semantic[:, :-2] * bin_logit.unsqueeze(-1)
                occ_conf = semantic[:, -2]
                confs.append(occ_conf)
            else:
                sem = semantic[:, :-1] * bin_logit.unsqueeze(-1)

            geo = 1 - bin_logit.unsqueeze(-1)

            if self.semantic_dim == 13:
                geosem = torch.cat([sem, geo], dim=-1)
            else:
                geosem = torch.cat([-10 * torch.ones_like(geo), sem, geo], dim=-1)

            semantics.append(geosem)
            densities.append(density)

            sem_labels.append(label[i])
            fov_mask.append(metas[i]['fov_mask'])

        result_dict.update(dict(
            depth_pred=anchor_depth,
            depth_gt=torch.stack([meta['depth_gt'] for meta in metas]),
        ))

        if len(semantics):
            semantics = torch.stack(semantics, dim=0).transpose(1, 2) # [1, 13, 129600]
            spatial_shape = label.shape[2:] # [60, 60, 36]

            result_dict.update({
                'ce_input': semantics.unflatten(-1, spatial_shape),
                'ce_label': torch.stack(sem_labels).flatten(0, 1),
                'fov_mask': torch.stack(fov_mask).bool(),
            })

            if self.with_unc:
                result_dict.update(dict(confs=confs))

        result_dict.update({
            'gaussian': gaussian,
        })

        return result_dict, None, None


@MODELS.register_module()
class VGGTGaussianSegmentorOnline(VGGTGaussianSegmentor):

    def __init__(
        self,
        *args,
        merge_kernel_dist_thresh=0.8,
        global_cuda_kwargs=None,
        opacities_threshold=0.1,
        freeze_local=False,
        detach_local=False,

        max_num_neighbors=3,
        opa_to_thickness=False,
        use_markley_avg_quat=False,
        use_log_avg=False,

        weight_use_p=False,
        weight_use_t=False,
        merge_scale_v2=False,


        use_diff_merge=False,
        diff_merge_cfg=None,
        train_global_pred=False,
        detach_global_each_frame=True,

        **kwargs,
    ):

        super().__init__(*args, opacities_threshold=opacities_threshold, **kwargs)
        self.global_gaussians = None
        self.merge_kernel_dist_thresh = merge_kernel_dist_thresh

        _dim_ = self.gs_pred_layer[0].in_features

        self.global_cuda_kwargs = global_cuda_kwargs
        from gpocc.model.head.gaussian_occ_head.ops.localagg_prob_gf2.local_aggregate_prob_gf2 import LocalAggregator
        self.global_aggregator = LocalAggregator(**global_cuda_kwargs)
        self.opacities_threshold = opacities_threshold

        self.detach_local = detach_local
        self.freeze_local = freeze_local

        self.max_num_neighbors = max_num_neighbors
        self.opa_to_thickness = opa_to_thickness
        self.use_markley_avg_quat = use_markley_avg_quat
        self.use_log_avg = use_log_avg
        self.weight_use_p = weight_use_p
        self.weight_use_t = weight_use_t
        self.merge_scale_v2 = merge_scale_v2
        
        self.use_diff_merge = use_diff_merge
        self.train_global_pred = train_global_pred
        self.detach_global_each_frame = detach_global_each_frame

        cfg = diff_merge_cfg or {}
        if self.use_diff_merge:
            self.diff_merge_tau = float(cfg.get("tau", 0.3))
            self.diff_merge_use_sem = bool(cfg.get("use_sem", True))
            self.diff_merge_alpha_init = float(cfg.get("alpha_init", 0.2))
            self.enable_birth = bool(cfg.get("enable_birth", False))
            self.birth_thresh = float(cfg.get("birth_thresh", 0.05))

            self.diff_updater = DiffGaussianUpdaterSparsePerA(
                tau=self.diff_merge_tau,
                use_sem=self.diff_merge_use_sem,
                alpha_init=self.diff_merge_alpha_init,
            )
        else:
            self.diff_updater = None

    def prepare_gaussian_args_v2(self, gaussians, metas):
        means = gaussians.means # b, g, 3
        b_, g_, _ = means.shape
        means_cam = F.pad(means, (0, 1), value=1)
        cam2world = torch.stack([meta['cam2world'] for meta in metas]).float().cuda()
        means_world_ = einsum(cam2world, means_cam, 'b n k, b j k -> b j n')
        means_world = means_world_[..., :3]
        means = means_world
        scales = gaussians.scales # b, g, 3
        rotations = gaussians.rotations # b, g, 4
        opacities = gaussians.semantics # b, g, c
        origi_opa = gaussians.opacities # b, g, 1

        if origi_opa.numel() == 0:
            origi_opa = torch.ones_like(opacities[..., :1], requires_grad=False)

        bs, g, _ = means.shape


        R = get_rotation_matrix(rotations) # b, g, 3, 3

        A = cam2world[:, :3, :3]                    # [b,3,3]
        A_T = A.transpose(-1, -2)                   # [b,3,3]
        A    = A.unsqueeze(1).expand(bs, g, 3, 3).contiguous().to(torch.float32)
        A_T  = A_T.unsqueeze(1).expand(bs, g, 3, 3).contiguous().to(torch.float32)

        R_world = torch.matmul(R.to(A_T.dtype), A_T)        # [b, g, 3, 3]


        return means, origi_opa, opacities, scales, R_world

    def forward(
        self,
        scenemeta=None,
        imgs=None,
        metas=None,
        points=None,
        label=None,
        frame_idx=0,
        grad_frames=False,
        test_mode=False,
        only_global=False,
    ):

        if self.training and self.detach_global_each_frame:
            self.detach_global_gaussians()

        result_dict, _, _ = super().forward(
            imgs=imgs,
            metas=metas,
            points=points,
            label=label,
            grad_frames=grad_frames,
            test_mode=test_mode,
            return_gaussian=only_global,
        )

        if not self.training:
            batch_size = len(metas)
            assert batch_size == 1
            batch_idx = 0

            nyu_pc_min = metas[batch_idx]['vox_origin'].cuda()
            scene_size = metas[batch_idx]['scene_size'].cuda()
            cam2world = metas[batch_idx]['cam2world'].cuda()
            nyu_pc_max = nyu_pc_min + scene_size

            sparse_means, sparse_origi_opa, sparse_opacities, sparse_scales, sparse_rots = self.prepare_gaussian_args_v2(
                result_dict['gaussian'], [metas[batch_idx]])
            sparse_qs = matrix_to_quaternion(sparse_rots)


            pos_mask = (sparse_origi_opa[batch_idx] > self.opacities_threshold).squeeze(1)

            result_dict['gaussian'] = GaussianPrediction(
                means=sparse_means[batch_idx][pos_mask][None],
                scales=sparse_scales[batch_idx][pos_mask][None],
                rotations=sparse_qs[batch_idx][pos_mask][None],
                opacities=sparse_origi_opa[batch_idx][pos_mask][None],
                semantics=sparse_opacities[batch_idx][pos_mask][None],
            )
        else:
            mask = None

        if frame_idx > 0:
            merge_fn = self.add_gaussian_nonlearnable
            global_gaussians = merge_fn(
                self.detach_gaussians(result_dict['gaussian']) if (self.detach_local and self.training) else result_dict['gaussian'],
                frame_idx=frame_idx,
                scenemetas=scenemeta,
                global_gaussians=self.global_gaussians,
            )
        else:
            global_gaussians = [result_dict['gaussian']]

        if (not self.training) or self.train_global_pred:
            global_result_dict = self.gaussian2pred(
                gaussians=global_gaussians,
                metas=[m[0] for m in scenemeta]
            )
        else:
            global_result_dict = None

        self.global_gaussians = global_gaussians
        return result_dict, global_result_dict

    def scene_init(self, scenemeta):
        self.scene_names = [m['scene_name'] for m in scenemeta]
        self.global_scene_dim = [m['global_scene_dim'] for m in scenemeta]
        self.global_scene_size = [m['global_scene_size'] for m in scenemeta]
        self.global_labels = [m['global_labels'] for m in scenemeta]
        self.global_xyz = [m['global_pts'].float() for m in scenemeta]
        self.global_scene_origin = [m['global_scene_origin'] for m in scenemeta]
        self.K_frams = [len(m['valid_img_paths']) for m in scenemeta]

        device = torch.device('cuda')
        self.global_mask_thistime = [
            torch.zeros_like(m['global_mask'], dtype=torch.bool, device=device)
            for m in scenemeta
        ]


    @torch.no_grad()
    def detach_gaussians(self, gaussian):
        new_gaussian = GaussianPrediction(
            means = gaussian.means.detach(),
            scales = gaussian.scales.detach(),
            rotations = gaussian.rotations.detach(),
            opacities = gaussian.opacities.detach(),
            semantics = gaussian.semantics.detach(),
            feat = gaussian.feat.detach() if gaussian.feat is not None else None,
            conf = gaussian.conf.detach() if gaussian.conf else None,
        )
        del gaussian
        return new_gaussian

    @torch.no_grad()
    def detach_global_gaussians(self):
        if self.global_gaussians is None:
            return
        for i, gaussian in enumerate(self.global_gaussians):
            self.global_gaussians[i] = self.detach_gaussians(gaussian)
            del gaussian

    def gaussian2pred(self, gaussians=None, metas=None):
        gaussians = gaussians if gaussians else self.global_gaussians
        batch_size = len(gaussians)

        nyu_pc_min = torch.stack([meta['global_scene_origin'] for meta in metas]).cuda()
        scene_size = torch.stack([meta['global_scene_size'] for meta in metas]).cuda()
        nyu_pc_max = nyu_pc_min + scene_size

        semantics = []
        densities = []
        sem_labels = []
        fov_mask = []
        confs = []

        for batch_idx in range(batch_size):
            gaussian = gaussians[batch_idx]


            means = gaussian.means
            origi_opa = gaussian.opacities
            opacities = gaussian.semantics
            scales = gaussian.scales
            rots = gaussian.rotations
            rots = quaternion_to_matrix(rots)

            S = torch.diag_embed(scales)
            M = torch.matmul(S.to(rots.dtype), rots)   # [b, g, 3, 3]  (= S @ R_world)
            Cov = torch.matmul(M.transpose(-1, -2), M)       # [b, g, 3, 3]
            CovInv = Cov.double().inverse().to(Cov.dtype)

            sampled_xyz = self.global_xyz[batch_idx].float()

            epsilon = 1e-3
            mask = (
                means[0, ..., 0] > (nyu_pc_min[batch_idx, 0]+epsilon)
            ) & (
                means[0, ..., 0] < (nyu_pc_max[batch_idx, 0]-epsilon)
            ) & (
                means[0, ..., 1] > (nyu_pc_min[batch_idx, 1]+epsilon)
            ) & (
                means[0, ..., 1] < (nyu_pc_max[batch_idx, 1]-epsilon)
            ) & (
                means[0, ..., 2] > (nyu_pc_min[batch_idx, 2]+epsilon)
            ) & (
                means[0, ..., 2] < (nyu_pc_max[batch_idx, 2]-epsilon)
            )

            if self.training and mask.sum() < 100:
                print(f'skip due to only {mask.sum()} points inside range')
                continue

            if not self.training and mask.sum() == 0:
                semantic = torch.zeros(
                    *sampled_xyz[None].shape[:4],
                    opacities.shape[-1],
                    device=opacities.device,
                    dtype=opacities.dtype).flatten(0, -2)
                bin_logit = torch.zeros_like(semantic[..., 0])
                density = torch.zeros_like(semantic[..., 0])

            else:
                semantic, bin_logit, density = self.global_aggregator(
                    sampled_xyz[None].flatten(1, 3),
                    means[0][mask][None],
                    origi_opa[0][mask][None],
                    opacities[0][mask][None],
                    scales[0][mask][None],
                    CovInv[0][mask][None],
                    metas[batch_idx],
                    nyu_pc_min[batch_idx]) # n, c


            sem = semantic[:, :-1] * bin_logit.unsqueeze(-1)
            geo = 1 - bin_logit.unsqueeze(-1)
            if self.semantic_dim == 13:
                geosem = torch.cat([sem, geo], dim=-1)
            else:
                geosem = torch.cat([-10 * torch.ones_like(geo), sem, geo], dim=-1)

            semantics.append(geosem.transpose(0, 1).unflatten(1, self.global_labels[batch_idx].shape))
            densities.append(density)
            cur_label = self.global_labels[batch_idx].detach().clone()
            cur_label[cur_label == 0] = 12
            sem_labels.append(cur_label)

            fov_mask.append(metas[batch_idx]['mask_in_global_from_this'] > 0)

        result_dict = dict()
        if len(semantics):
            result_dict.update({
                'ce_input': semantics,
                'ce_label': sem_labels,
                'fov_mask': fov_mask,
            })

        return result_dict

    def update_global_mask(self, scenemetas, frame_idx=0):
        mask_in_global_from_this = [meta[frame_idx]['mask_in_global_from_this'].to(dtype=torch.bool).cuda() for meta in scenemetas]
        self.global_mask_thistime = [a | b for a, b in zip(self.global_mask_thistime, mask_in_global_from_this)]

    def get_global_gaussian(self):
        return self.global_gaussians

    def add_gaussian_nonlearnable(self, gaussian=None, frame_idx=0, scenemetas=None, global_gaussians=None, mask=None):

        dist_thresh = self.merge_kernel_dist_thresh

        if radius is None:
            print('torch_cluster is needed for this, install torch_cluster first')
            raise torch_cluster_import_e

        if torch_scatter is None:
            print('torch_scatter is needed for this, install torch_cluster first')
            raise torch_scatter_import_e

        fused_gaussians = []

        means = gaussian.means
        device = means.device
        dtype = means.dtype
        batch_size = means.shape[0]

        for batch_idx in range(batch_size):

            xyz_A = global_gaussians[batch_idx].means.squeeze(0)
            conf_A = global_gaussians[batch_idx].conf.squeeze(0) if global_gaussians[batch_idx].conf else None
            sem_A = global_gaussians[batch_idx].semantics.squeeze(0)
            scale_A = global_gaussians[batch_idx].scales.squeeze(0)
            rot_A   = global_gaussians[batch_idx].rotations.squeeze(0)
            opa_A   = global_gaussians[batch_idx].opacities.squeeze(0)

            pos_mask = (gaussian.opacities[batch_idx] > self.opacities_threshold).squeeze(1)

            xyz_B = gaussian.means[batch_idx][pos_mask]
            conf_B = gaussian.conf[batch_idx][pos_mask] if gaussian.conf else None
            sem_B = gaussian.semantics[batch_idx][pos_mask]
            scale_B = gaussian.scales[batch_idx][pos_mask]
            rot_B   = gaussian.rotations[batch_idx][pos_mask]
            opa_B   = gaussian.opacities[batch_idx][pos_mask]

            xyz_B_in_A = xyz_B
            pos_A = xyz_A.view(-1, 3)  # [N, 3]
            pos_B = xyz_B.view(-1, 3)  # [M, 3]

            idx_A, idx_B = radius(pos_B, pos_A, r=dist_thresh, max_num_neighbors=self.max_num_neighbors)

            def conf_act(c):
                return 1 + c.exp()

            xyz_b = pos_B[idx_B]
            sem_b = sem_B[idx_B]
            merged_A_idx = torch.unique(idx_A)

            if conf_B is None:
                assert conf_A is None
                conf_b = torch.ones_like(xyz_B[:, 0:1])[idx_B]
                conf_a = torch.ones_like(xyz_A[:, 0:1])[merged_A_idx]
                num_b = torch_scatter.scatter_add(torch.ones_like(conf_b), idx_A, dim=0, dim_size=xyz_A.size(0))
            else:
                import pdb; pdb.set_trace()
                conf_b = conf_act(conf_B[idx_B])    # [K]
                num_b = torch_scatter.scatter_add(torch.ones_like(conf_b), idx_A, dim=0, dim_size=xyz_A.size(0))
                merged_conf = conf_sum / (num_b + 1)
                merged_conf = merged_conf[merged_A_idx]
                conf_a = conf_A[idx_A]

            def _pmax_weight_from_logits(logits, T=1.0, alpha=1.0, floor=None):
                """
                logits: [..., C] 语义logits
                返回:   [..., 1]  权重(0~1)，Top-1 概率经温度缩放 + 可选阈值拉伸 + 幂
                """
                p = F.softmax(logits / T, dim=-1)
                pmax, _ = p.max(dim=-1, keepdim=True)         # [..., 1]
                if floor is not None:
                    pmax = (pmax - floor).clamp(min=0.0) / (1.0 - floor + 1e-8)
                if alpha != 1.0:
                    pmax = pmax.pow(alpha)
                return pmax.clamp(0.0, 1.0)

            if self.weight_use_p:
                prob_B = _pmax_weight_from_logits(sem_B[idx_B])
                prob_A = _pmax_weight_from_logits(sem_A[merged_A_idx])

                conf_b = conf_b * prob_B
                conf_a = conf_a * prob_A

            if self.weight_use_t:
                gamma = float(getattr(self, "temporal_alpha", 0.1))  # =EMA的γ
                conf_a = conf_a * gamma            # 旧(A)占γ
                conf_b = conf_b * (1.0 - gamma)    # 新(B)占(1-γ)

            conf_sum = torch_scatter.scatter_add(
                conf_b, idx_A, dim=0, dim_size=xyz_A.size(0)) + 1e-6
            conf_sum = conf_sum[merged_A_idx] + conf_a

            weighted_xyz_b = xyz_b * conf_b
            agg_xyz = torch_scatter.scatter_add(
                weighted_xyz_b, idx_A, dim=0, dim_size=xyz_A.size(0))
            merged_xyz = (agg_xyz[merged_A_idx] + xyz_A[merged_A_idx] * conf_a) / conf_sum

            weighted_sem_b = sem_b * conf_b
            agg_sem = torch_scatter.scatter_add(
                weighted_sem_b, idx_A, dim=0, dim_size=sem_A.size(0))
            merged_sem = (agg_sem[merged_A_idx] + sem_A[merged_A_idx] * conf_a) / conf_sum


            weighted_scale_b = scale_B[idx_B] * conf_b
            agg_scale = torch_scatter.scatter_add(weighted_scale_b, idx_A, dim=0, dim_size=scale_A.size(0))
            merged_scale = (agg_scale[merged_A_idx] + scale_A[merged_A_idx] * conf_a) / conf_sum

            weighted_opa_b = opa_B[idx_B] * conf_b
            agg_opa = torch_scatter.scatter_add(weighted_opa_b, idx_A, dim=0, dim_size=opa_A.size(0))
            merged_opa = (agg_opa[merged_A_idx] + opa_A[merged_A_idx] * conf_a) / conf_sum

            q_ref_pairs = rot_A[idx_A]
            q_b = rot_B[idx_B]
            dot = (q_b * q_ref_pairs).sum(dim=-1, keepdim=True)
            q_b_aligned = torch.where(dot < 0, -q_b, q_b)

            weighted_q_b = q_b_aligned * conf_b
            agg_q = torch_scatter.scatter_add(weighted_q_b, idx_A, dim=0, dim_size=rot_A.size(0))
            eps = 1e-9
            merged_rot = agg_q[merged_A_idx] + rot_A[merged_A_idx] * conf_a
            merged_rot = merged_rot / (merged_rot.norm(dim=-1, keepdim=True) + eps)

            used_B_idx = torch.unique(idx_B)
            keep_A = torch.ones(xyz_A.shape[0], dtype=torch.bool, device=xyz_A.device)
            keep_B = torch.ones(xyz_B.shape[0], dtype=torch.bool, device=xyz_B.device)
            keep_A[merged_A_idx] = False
            keep_B[used_B_idx] = False

            final_xyz = torch.cat([merged_xyz, xyz_A[keep_A], xyz_B_in_A[keep_B]], dim=0)[None]  # [1, N, 3]
            final_sem = torch.cat([merged_sem, sem_A[keep_A], sem_B[keep_B]], dim=0)[None]  # [1, N, C]
            final_scale = torch.cat([merged_scale, scale_A[keep_A], scale_B[keep_B]], dim=0)[None]
            final_opa = torch.cat([merged_opa, opa_A[keep_A], opa_B[keep_B]], dim=0)[None]
            final_rot = torch.cat([merged_rot, rot_A[keep_A], rot_B[keep_B]], dim=0)[None]

            fused_gaussian = GaussianPrediction(
                means=final_xyz,
                scales=final_scale,
                rotations=final_rot,
                opacities=final_opa,
                semantics=final_sem,
            )
            fused_gaussians.append(fused_gaussian)

        return fused_gaussians
