# GPOcc Demo

Single-image semantic occupancy prediction with GPOcc. Give it one RGB image and it predicts a 60×60×36 occupancy grid (8 cm voxels, a 4.8 m × 4.8 m × 2.88 m box). The occupied voxel centers are saved as a colored point cloud (`.ply`).

This folder is self-contained. It only needs PyTorch and a few small packages: no mmcv/mmdet/mmengine, pytorch3d, or torch-scatter. The CUDA op is compiled automatically the first time you load a model.

## Installation

```bash
cd demo
pip install -r requirements.txt
```

You also need PyTorch with CUDA and the CUDA toolkit (`nvcc`). The toolkit version must match the CUDA version of your PyTorch build. If `nvcc` is not on your `PATH`, set `CUDA_HOME`:

```bash
export CUDA_HOME=/usr/local/cuda-12.1
```

The first time a model is loaded, the Gaussian-to-voxel CUDA op is compiled (about 1–2 min). The result is cached in `~/.cache/torch_extensions`, so later runs start immediately.

## Models

Both models are in one Hugging Face repo, one subfolder each:

| `--variant` / `subfolder` | Backbone | Params | Size |
|---|---|---|---|
| `vggt` (default) | VGGT-1B aggregator | 942M | 3.8 GB |
| `dpt` | Depth-Anything-V2 ViT-B | 98M | 0.4 GB |

`--model` accepts a Hugging Face repo id or a local folder with the same layout.

## Quick Start

The example images in `examples/` come from the ScanNet test split. Each image has a JSON file with its camera parameters:

```bash
python demo.py --model JuIvyy/GPOcc --image examples/scene0000_00_00061.jpg
# use the lighter model: add --variant dpt
# -> outputs/scene0000_00_00061_occ.ply   colored occupied voxel centers
```

> **About the output.** The `.ply` file is **not** a reconstructed mesh, a depth point cloud, or the predicted Gaussians. It contains **one point per occupied voxel**, placed at the **voxel center**. Empty voxels are not saved, so the points form a regular 3D grid with 8 cm spacing. Each point is colored by its predicted class. Coordinates are in meters, in the world frame given by `cam2world` (z up).

Open the `.ply` file in MeshLab, CloudCompare, or Open3D. Colors per class:

| class | color | class | color |
|---|---|---|---|
| ceiling | ![](https://placehold.co/12x12/16bfce/16bfce.png) | bed | ![](https://placehold.co/12x12/cccc5b/cccc5b.png) |
| floor | ![](https://placehold.co/12x12/d62628/d62628.png) | sofa | ![](https://placehold.co/12x12/ffba77/ffba77.png) |
| wall | ![](https://placehold.co/12x12/2ba02b/2ba02b.png) | table | ![](https://placehold.co/12x12/9366bc/9366bc.png) |
| window | ![](https://placehold.co/12x12/9ed8e5/9ed8e5.png) | tvs | ![](https://placehold.co/12x12/1e77b5/1e77b5.png) |
| chair | ![](https://placehold.co/12x12/729ece/729ece.png) | furniture | ![](https://placehold.co/12x12/bcbc21/bcbc21.png) |
| | | objects | ![](https://placehold.co/12x12/ff7f0c/ff7f0c.png) |

## Using Your Own Images

```bash
python demo.py --model JuIvyy/GPOcc --image my_room.jpg
```

Camera parameters are read from `--camera cam.json`. If that is not given, the demo looks for `<image>.json` next to the image. Every field is optional:

```json
{
  "intrinsics": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
  "cam2world": [[...4x4...]],
  "vox_origin": [x, y, z]
}
```

- `intrinsics`: the camera matrix at the original image resolution. If missing, the demo assumes a ScanNet-like field of view (`fx = fy = 0.9 * width`). Use real intrinsics when you have them, because depth scale depends on them.
- `cam2world`: the camera pose in a gravity-aligned, z-up world frame. It only decides where the output grid is placed and how it is oriented. If missing, the camera is placed `--cam-height` meters (default 1.5) above the floor and tilted down by `--pitch` degrees (default 25, the median tilt in ScanNet; use `--pitch 0` for level photos).
- `vox_origin`: the world position of the grid corner. If missing, the 4.8 m box is placed the way OccScanNet does it: the bottom sits on the floor and the box extends forward from the camera.

Add `--fov-only` to keep only the voxels that are visible in the image.

## Python API

```python
from gpocc_hub import GPOcc, CLASS_NAMES

model = GPOcc.from_pretrained("JuIvyy/GPOcc").cuda().eval()                    # VGGT
# model = GPOcc.from_pretrained("JuIvyy/GPOcc", subfolder="dpt").cuda().eval()  # DPT
out = model.predict("examples/scene0000_00_00061.jpg", intrinsics=K)  # K is optional

out["occupancy"]      # (60, 60, 36) class index per voxel, 12 = empty, see CLASS_NAMES
out["voxel_centers"]  # (60, 60, 36, 3) world coordinates
out["fov_mask"]       # (60, 60, 36) voxels inside the camera frustum
out["gaussians"]      # predicted 3D Gaussians (means/scales/rotations/opacities/semantics), camera frame
```

For lower-level access, `model.predict_gaussians(images, K)` returns the Gaussians for a batch, and `model.voxelize(gaussians, cam2world, vox_origin)` returns the 13-class occupancy logits.

## License

The model weights are derived from VGGT-1B and Depth-Anything-V2-Base, and are released for non-commercial research use only. The Gaussian aggregation CUDA op comes from Inria's GRAPHDECO code (non-commercial license). The example images are from [ScanNet](http://www.scan-net.org/) and are subject to the ScanNet Terms of Use.
