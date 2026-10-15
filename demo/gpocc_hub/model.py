"""Self-contained GPOcc inference model (no mm* dependencies).

Usage:
    from gpocc_hub import GPOcc
    model = GPOcc.from_pretrained("JuIvyy/GPOcc").cuda().eval()
    out = model.predict("room.jpg", intrinsics=K)
"""
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
from huggingface_hub import PyTorchModelHubMixin

from . import ops
from .utils import (CLASS_NAMES, EMPTY_IDX, default_cam2world, default_intrinsics,
                    default_vox_origin, get_rotation_matrix, resize_for_model, safe_sigmoid)


class _VGGTEncoder(nn.Module):
    """VGGT with only the aggregator kept (camera / depth / point / track heads dropped)."""

    def __init__(self, img_size=518, patch_size=14, embed_dim=1024):
        super().__init__()
        from .vggt.models.aggregator import Aggregator
        self.aggregator = Aggregator(img_size=img_size, patch_size=patch_size, embed_dim=embed_dim)


class GPOcc(
    nn.Module,
    PyTorchModelHubMixin,
    library_name='gpocc',
    tags=['occupancy-prediction', '3d-gaussians', 'monocular', 'cvpr2026'],
    license='cc-by-nc-4.0',
    pipeline_tag='image-to-3d',
):
    """Monocular semantic occupancy prediction with sparse 3D Gaussians.

    Args:
        backbone: 'vggt' (VGGT-1B aggregator) or 'dpt' (Depth-Anything-V2 ViT-B).
        semantic_dim: number of semantic channels per Gaussian.
        num_bins: number of Gaussians sampled along each anchor ray.
        grid_shape / voxel_size: output occupancy grid (default 60x60x36 @ 8cm).
    """

    def __init__(
        self,
        backbone='vggt',
        semantic_dim=12,
        num_bins=16,
        scale_range=(0.01, 0.08),
        opacities_threshold=0.01,
        max_depth=8.0,
        grid_shape=(60, 60, 36),
        voxel_size=0.08,
        scale_multiplier=3,
    ):
        super().__init__()
        self.backbone_type = backbone
        self.semantic_dim = semantic_dim
        self.num_bins = num_bins
        self.scale_range = tuple(scale_range)
        self.opacities_threshold = opacities_threshold
        self.max_depth = max_depth
        self.grid_shape = tuple(grid_shape)
        self.voxel_size = voxel_size
        self.scene_size = tuple(s * voxel_size for s in grid_shape)

        if backbone == 'vggt':
            from .vggt.heads.dpt_head import DPTHead
            self.backbone = _VGGTEncoder()
            self.gs_head = DPTHead(dim_in=2 * 1024, output_dim=2, activation='exp', conf_activation='expp1')
            feat_dim = 128
        elif backbone == 'dpt':
            from .depth_anything_v2.dpt import DepthAnythingV2
            self.backbone = DepthAnythingV2(
                encoder='vitb', features=128, out_channels=[96, 192, 384, 768], max_depth=20)
            self.gs_head = self.backbone.depth_head
            self.backbone.depth_head = None
            feat_dim = 64
        else:
            raise ValueError(f'Unknown backbone: {backbone}')

        self.aggregator = ops.LocalAggregator(
            scale_multiplier=scale_multiplier,
            H=grid_shape[0], W=grid_shape[1], D=grid_shape[2],
            pc_min=[-51.2, -51.2, -5.0],
            grid_size=voxel_size)

        _dim_ = 256
        self.gs_pred_layer = nn.Sequential(
            nn.Linear(_dim_, _dim_),
            nn.ReLU(),
            nn.Linear(_dim_, semantic_dim + 3 + 4 + 1),
        )
        self.anchor_feat_layer = nn.Sequential(
            nn.Conv2d(feat_dim, _dim_, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(_dim_, _dim_, kernel_size=2, stride=2),
        )
        self.anchor_depth_pred = nn.Sequential(
            nn.Conv2d(_dim_, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0),
        )
        if num_bins > 0:
            self.bins_emb = nn.Embedding(num_bins, _dim_)
            self.depth_scale_layer = nn.Conv2d(_dim_, 1, kernel_size=1)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *, subfolder=None, **kwargs):
        """Load from a local folder or a Hugging Face repo.

        One repo can hold several models in subfolders (e.g. `vggt/`, `dpt/`):
            GPOcc.from_pretrained("JuIvyy/GPOcc")                   # -> vggt/
            GPOcc.from_pretrained("JuIvyy/GPOcc", subfolder="dpt")
        """
        path = _resolve_subfolder(str(pretrained_model_name_or_path), subfolder, kwargs)
        model = super().from_pretrained(path, **kwargs)
        # Build (or load the cached) CUDA op right away so the first predict() is fast.
        if torch.cuda.is_available():
            ops.try_load_ext()
        return model

    # ------------------------------------------------------------------ #
    # Gaussian prediction (camera frame)
    # ------------------------------------------------------------------ #
    def _image_features(self, imgs):
        """imgs: (B, 1, 3, H, W) BGR in [0, 1] -> (B, C, H/2, W/2) features."""
        if self.backbone_type == 'vggt':
            tokens, patch_start_idx = self.backbone.aggregator(images=imgs)
            feat = self.gs_head.custom_forward(tokens, images=imgs, patch_start_idx=patch_start_idx)
        else:
            feat = self.gs_head.custom_forward(*self.backbone.custom_forward(imgs.flatten(0, 1)))
            feat = feat.unflatten(0, imgs.shape[:2])
        return feat.flatten(0, 1)

    def predict_gaussians(self, images, intrinsics):
        """
        Args:
            images: (B, 3, H, W) RGB in [0, 1]; H, W multiples of 14 (long side 518 recommended).
            intrinsics: (B, 3, 3) for the given image resolution.
        Returns:
            dict of Gaussians in the camera frame (x right, y down, z forward):
            means (B, G, 3), scales (B, G, 3), rotations (B, G, 4, wxyz),
            opacities (B, G, 1), semantics (B, G, semantic_dim).
        """
        # The released models were trained on BGR inputs.
        imgs = images.flip(1)[:, None]

        anchor_feat = self.anchor_feat_layer(self._image_features(imgs))
        anchor_depth = self.anchor_depth_pred(anchor_feat).sigmoid() * self.max_depth

        device, dtype = anchor_depth.device, anchor_depth.dtype
        B, _, H, W = anchor_depth.shape
        iH, iW = imgs.shape[-2:]

        u, v = torch.meshgrid(
            torch.arange(W, device=device), torch.arange(H, device=device), indexing='xy')
        uv = torch.stack([u, v, torch.ones_like(u)], dim=0).float()
        uv = uv.unsqueeze(0).repeat(B, 1, 1, 1).to(dtype)

        intr = intrinsics.to(dtype=dtype, device=device)
        fx = intr[:, 0, 0].view(B, 1, 1)
        fy = intr[:, 1, 1].view(B, 1, 1)
        cx = intr[:, 0, 2].view(B, 1, 1)
        cy = intr[:, 1, 2].view(B, 1, 1)

        x = (uv[:, 0] / W * iW - cx) / fx
        y = (uv[:, 1] / H * iH - cy) / fy
        dirs = torch.stack([x, y, torch.ones_like(x)], dim=1)
        dirs = torch.nn.functional.normalize(dirs, dim=1, eps=1e-6)
        xyz = dirs * anchor_depth

        if self.num_bins > 0:
            depth_scale = self.depth_scale_layer(anchor_feat).sigmoid()
            depth_offset = torch.linspace(
                0, 5, self.num_bins, device=device, dtype=dtype)[None, :, None, None] * depth_scale
            dirs = dirs.unsqueeze(1).expand(-1, self.num_bins, -1, -1, -1)
            xyz = xyz.unsqueeze(1) + dirs * depth_offset.unsqueeze(2)
            feat = anchor_feat.unsqueeze(1) + self.bins_emb.weight[None, :, :, None, None]
        else:
            xyz = xyz[:, None]
            feat = anchor_feat[:, None]

        xyz = rearrange(xyz, 'b n c h w -> b (n h w) c')
        feat = rearrange(feat, 'b n c h w -> b (n h w) c')
        gs_feat = self.gs_pred_layer(feat)

        scales = safe_sigmoid(gs_feat[..., :3])
        scales = self.scale_range[0] + (self.scale_range[1] - self.scale_range[0]) * scales
        return dict(
            means=xyz,
            scales=scales,
            rotations=gs_feat[..., 3:7],
            opacities=safe_sigmoid(gs_feat[..., 7:8]),
            semantics=gs_feat[..., 8:8 + self.semantic_dim],
        )

    # ------------------------------------------------------------------ #
    # Voxelization (world frame)
    # ------------------------------------------------------------------ #
    def voxel_centers(self, vox_origin):
        """vox_origin: (B, 3) -> (B, X, Y, Z, 3) voxel centers in world frame."""
        idx = torch.stack(torch.meshgrid(
            *[torch.arange(n, device=vox_origin.device) for n in self.grid_shape],
            indexing='ij'), dim=-1).float()
        return vox_origin[:, None, None, None].float() + (idx + 0.5) * self.voxel_size

    @torch.no_grad()
    def voxelize(self, gaussians, cam2world, vox_origin):
        """
        Args:
            gaussians: output of `predict_gaussians`.
            cam2world: (B, 4, 4) camera-to-world transform of a gravity-aligned (z-up) frame.
            vox_origin: (B, 3) world coordinate of the grid corner (voxel (0, 0, 0)).
        Returns:
            logits (B, 13, X, Y, Z): channel 0 'unknown' (never predicted), 1-11 semantic
            classes, 12 'empty'. See `CLASS_NAMES`.
        """
        means = gaussians['means']
        device = means.device
        B, G, _ = means.shape
        cam2world = cam2world.to(device).float()
        vox_origin = vox_origin.to(device).float()
        pc_max = vox_origin + torch.tensor(self.scene_size, device=device)
        centers = self.voxel_centers(vox_origin)

        means_world = torch.einsum('bnk,bjk->bjn', cam2world, nn.functional.pad(means, (0, 1), value=1))[..., :3]
        scales = gaussians['scales']
        S = torch.diag_embed(scales)
        R = get_rotation_matrix(gaussians['rotations'])
        M = S @ R
        cov = M.transpose(-1, -2) @ M
        rot = cam2world[:, None, :3, :3]
        cov = rot @ cov @ rot.transpose(-1, -2)
        cov_inv = cov.double().inverse().to(cov.dtype)
        opas = gaussians['opacities']
        semantics = gaussians['semantics']

        logits = []
        eps = 1e-3
        for i in range(B):
            mask = ((means_world[i] > vox_origin[i] + eps) & (means_world[i] < pc_max[i] - eps)).all(-1)
            mask = mask & (opas[i].squeeze(-1) > self.opacities_threshold)
            pts = centers[i].flatten(0, 2)
            if mask.sum() == 0:
                semantic = torch.zeros(pts.shape[0], self.semantic_dim, device=device)
                bin_logit = torch.zeros_like(semantic[:, 0])
            else:
                with torch.cuda.device(device):
                    semantic, bin_logit, _ = self.aggregator(
                        pts, means_world[i][mask], opas[i][mask], semantics[i][mask],
                        scales[i][mask], cov_inv[i][mask], vox_origin[i])
            sem = semantic[:, :-1] * bin_logit.unsqueeze(-1)
            geo = 1 - bin_logit.unsqueeze(-1)
            if self.semantic_dim == len(CLASS_NAMES):
                geosem = torch.cat([sem, geo], dim=-1)
            else:
                geosem = torch.cat([-10 * torch.ones_like(geo), sem, geo], dim=-1)
            logits.append(geosem.T.unflatten(-1, self.grid_shape))
        return torch.stack(logits)

    def forward(self, images, intrinsics, cam2world, vox_origin):
        gaussians = self.predict_gaussians(images, intrinsics)
        return self.voxelize(gaussians, cam2world, vox_origin), gaussians

    # ------------------------------------------------------------------ #
    # One-call API
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def predict(self, image, intrinsics=None, cam2world=None, vox_origin=None,
                pitch_deg=25.0, cam_height=1.5):
        """Predict semantic occupancy from a single image.

        Args:
            image: file path, PIL.Image, or HxWx3 uint8 RGB numpy array.
            intrinsics: 3x3 camera matrix at the original image resolution.
                Defaults to a ScanNet-like FoV (fx = 0.9 * width) if omitted.
            cam2world: 4x4 camera-to-world pose in a gravity-aligned, z-up frame.
                Defaults to a camera `cam_height` m above the floor, tilted down
                by `pitch_deg` (the median tilt in ScanNet; use 0 for level shots).
            vox_origin: (3,) world position of the occupancy grid corner. Defaults
                to the OccScanNet convention: box on the floor, in front of the camera.
        Returns:
            dict with
              occupancy (X, Y, Z) long: class index per voxel, 12 = empty
              logits (13, X, Y, Z), fov_mask (X, Y, Z) bool, voxel_centers (X, Y, Z, 3),
              gaussians (camera frame), cam2world, vox_origin, intrinsics (resized), image (BGR float)
        """
        image_rgb = _load_rgb(image)
        H, W = image_rgb.shape[:2]
        if intrinsics is None:
            intrinsics = default_intrinsics(H, W)
        if cam2world is None:
            cam2world = default_cam2world(pitch_deg, cam_height)
        if vox_origin is None:
            vox_origin = default_vox_origin(cam2world, self.scene_size, cam_height)

        image_bgr, K = resize_for_model(image_rgb[..., ::-1], intrinsics)
        device = next(self.parameters()).device
        img = torch.from_numpy(np.ascontiguousarray(image_bgr[..., ::-1])).permute(2, 0, 1)[None] / 255.
        K_t = torch.from_numpy(K).float()[None].to(device)
        T = torch.as_tensor(np.asarray(cam2world), dtype=torch.float32)[None].to(device)
        origin = torch.as_tensor(np.asarray(vox_origin), dtype=torch.float32)[None].to(device)

        logits, gaussians = self(img.to(device), K_t, T, origin)
        centers = self.voxel_centers(origin)[0]
        return dict(
            occupancy=logits[0].argmax(0),
            logits=logits[0],
            fov_mask=_fov_mask(centers, T[0], K_t[0], image_bgr.shape[:2]),
            voxel_centers=centers,
            gaussians={k: v[0] for k, v in gaussians.items()},
            cam2world=T[0],
            vox_origin=origin[0],
            intrinsics=K_t[0],
            image=image_bgr,
        )


DEFAULT_SUBFOLDER = 'vggt'


def _resolve_subfolder(name, subfolder, hub_kwargs):
    """Map (repo or folder, subfolder) to a path that PyTorchModelHubMixin can load."""
    import os.path as osp

    if osp.isdir(name):
        if subfolder is None and not osp.exists(osp.join(name, 'config.json')):
            subfolder = DEFAULT_SUBFOLDER
        return osp.join(name, subfolder) if subfolder else name

    from huggingface_hub import file_exists, snapshot_download

    download_kwargs = {k: hub_kwargs.pop(k) for k in
                       ['revision', 'cache_dir', 'token', 'force_download', 'local_files_only']
                       if k in hub_kwargs}
    if subfolder is None:
        offline = download_kwargs.get('local_files_only')
        if not offline and file_exists(name, 'config.json', revision=download_kwargs.get('revision'),
                                       token=download_kwargs.get('token')):
            return name  # single-model repo with weights at the root
        subfolder = DEFAULT_SUBFOLDER
    local = snapshot_download(name, allow_patterns=[f'{subfolder}/*'], **download_kwargs)
    return osp.join(local, subfolder)


def _load_rgb(image):
    from PIL import Image, ImageOps
    if isinstance(image, str):
        image = ImageOps.exif_transpose(Image.open(image))
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert('RGB'))
    image = np.asarray(image)
    assert image.ndim == 3 and image.shape[-1] == 3, 'expected an HxWx3 RGB image'
    return image


def _fov_mask(centers, cam2world, K, hw):
    pts = centers.reshape(-1, 3)
    world2cam = torch.linalg.inv(cam2world)
    cam = pts @ world2cam[:3, :3].T + world2cam[:3, 3]
    z = cam[:, 2]
    uv = (cam @ K.T)[:, :2] / z.clamp(min=1e-6)[:, None]
    H, W = hw
    mask = (z > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < W) & (uv[:, 1] >= 0) & (uv[:, 1] < H)
    return mask.reshape(centers.shape[:3])
