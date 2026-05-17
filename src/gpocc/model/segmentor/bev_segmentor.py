import torch
import numpy as np
from copy import deepcopy
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from mmseg.registry import MODELS as MODELS_SEG


@MODELS.register_module()
class BEVSegmentor(BaseModule):

    def __init__(
        self,
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
        if backbone is not None:
            self.backbone = MODELS.build(backbone)
        if neck is not None:
            try:
                self.neck = MODELS.build(neck)
            except:
                self.neck = MODELS_SEG.build(neck)
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
        imgs = imgs.reshape(B * N, C, H, W)
        img_feats_backbone, _ = self.backbone(imgs, use_image=True, use_points=False)
        if isinstance(img_feats_backbone, dict):
            img_feats_backbone = list(img_feats_backbone.values())
        img_feats = self.neck(img_feats_backbone)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped
    
    def obtain_bev(self, imgs, metas):
        B, F, N, C, H, W = imgs.shape
        imgs = imgs.reshape(B*F, N, C, H, W)
        mlvl_img_feats = self.extract_img_feat(imgs)
        bev_query = self.lifter(mlvl_img_feats)
        bev = self.encoder(bev_query, mlvl_img_feats, metas)
        return bev
    
    def forward(self,
                imgs=None,
                metas=None,
                points=None,
                label=None,
                grad_frames=None,
                test_mode=False,
                **kwargs,
        ):
        B, F, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'
        if grad_frames is not None:
            assert grad_frames < F
            imgs_grad, metas_grad, imgs_no_grad, metas_no_grad, inv_index = self.frame_split(grad_frames, imgs, metas)
            bev_grad = self.obtain_bev(imgs_grad, metas_grad)
            with torch.no_grad():
                bev_no_grad = self.obtain_bev(imgs_no_grad, metas_no_grad)
            bev = torch.cat([bev_grad, bev_no_grad], dim=0)[inv_index]
        else:
            bev = self.obtain_bev(imgs, metas)

        BF, H, W, C = bev.shape
        bev = bev.reshape(B, F, H, W, C)
        if hasattr(self, 'future_decoder'):
            output_dict = self.future_decoder(bev, metas)
            bev_predict = output_dict.pop('bev')
        else:
            bev_predict = bev
            output_dict = dict()
        output_dict = self.head(bev_feat=bev_predict, points=points, label=label, output_dict=output_dict)

        return output_dict
    
    def debug_l2(self, points_labels, fuse, prev, next, verbose=True):
        points = points_labels[0, -1, :, :3]
        labels = points_labels[0, -1, :, -1]
        fuse = fuse[0, -1]
        prev = prev[0, -1]
        next = next[0, -1]

        ### filtering
        norm = torch.norm(points, 2, dim=-1)
        grid_size = [100, 100, 8]
        pc = self.future_decoder.pc_range
        points[:, 0] = (points[:, 0] - pc[0]) / (pc[3] - pc[0]) * grid_size[0]
        points[:, 1] = (points[:, 1] - pc[1]) / (pc[4] - pc[1]) * grid_size[1]
        points[:, 2] = (points[:, 2] - pc[2]) / (pc[5] - pc[2]) * grid_size[2]
        points = points.long()
        mask = (points[:, 0] >= 0) & (points[:, 0] < grid_size[0]) & \
               (points[:, 1] >= 0) & (points[:, 1] < grid_size[1]) & \
               (points[:, 2] >= 0) & (points[:, 2] < grid_size[2]) & (norm > 1.0)
        points = points[mask]
        labels = labels[mask]

        ### sampling
        fuse_sample_all = fuse[points[:, 0], points[:, 1]]
        prev_sample_all = prev[points[:, 0], points[:, 1]]
        next_sample_all = next[points[:, 0], points[:, 1]]

        ### error
        error_prev = ((fuse_sample_all - prev_sample_all) ** 2).mean()
        error_next = ((fuse_sample_all - next_sample_all) ** 2).mean()

        if getattr(self, 'errors_prev', None) is None:
            self.errors_prev = []
        self.errors_prev.append(error_prev.item())
        if getattr(self, 'errors_next', None) is None:
            self.errors_next = []
        self.errors_next.append(error_next.item())

        if verbose:
            print(error_prev.item(), error_next.item())

    def save_debug(self,):
        assert len(self.errors_prev) > 0 and len(self.errors_next) > 0
        np.save('debug_l2_0.5gtpose.npy', np.array([self.errors_prev, self.errors_next]))
    
    def frame_split(self, grad_frames, imgs, metas):
        F = imgs.shape[1]
        index = np.random.permutation(F)
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
        B, F, N, C, H, W = imgs.shape
        assert B==1, 'bs > 1 not supported'
        
        bev = self.obtain_bev(imgs, metas)
        BF, H, W, C = bev.shape
        bev = bev.reshape(B, F, H, W, C)

        output_dict = self.future_decoder.forward_autoreg(bev, metas)
        bev_predict = output_dict.pop('bev')
        output_dict = self.head(bev_feat=bev_predict, points=points, label=label, output_dict=output_dict)

        return output_dict