"""Single-image GPOcc demo: image -> semantic occupancy -> colored voxel point cloud (.ply).

    python demo.py --model JuIvyy/GPOcc --image examples/scene0000_00_00061.jpg
"""
import argparse
import json
import os
import os.path as osp

import numpy as np
import torch

from gpocc_hub import CLASS_NAMES, EMPTY_IDX, GPOcc

# RGB color per class index (0 'unknown' and 12 'empty' are never written).
PALETTE = np.array([
    [0, 0, 0],        # unknown
    [22, 191, 206],   # ceiling
    [214, 38, 40],    # floor
    [43, 160, 43],    # wall
    [158, 216, 229],  # window
    [114, 158, 206],  # chair
    [204, 204, 91],   # bed
    [255, 186, 119],  # sofa
    [147, 102, 188],  # table
    [30, 119, 181],   # tvs
    [188, 188, 33],   # furniture
    [255, 127, 12],   # objects
    [255, 255, 255],  # empty
], dtype=np.uint8)


def save_ply(path, points, colors):
    vertex = np.empty(len(points), dtype=[
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    vertex['x'], vertex['y'], vertex['z'] = points.T
    vertex['red'], vertex['green'], vertex['blue'] = colors.T
    header = (
        'ply\nformat binary_little_endian 1.0\n'
        f'element vertex {len(points)}\n'
        'property float x\nproperty float y\nproperty float z\n'
        'property uchar red\nproperty uchar green\nproperty uchar blue\n'
        'end_header\n')
    with open(path, 'wb') as f:
        f.write(header.encode())
        f.write(vertex.tobytes())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='JuIvyy/GPOcc', help='local folder or Hugging Face repo id')
    parser.add_argument('--variant', default='vggt', choices=['vggt', 'dpt'], help='model subfolder in the repo')
    parser.add_argument('--image', required=True)
    parser.add_argument('--camera', default=None,
                        help='JSON with "intrinsics" (3x3) and optionally "cam2world" (4x4) / "vox_origin" (3); '
                             'defaults to <image>.json next to the image if present')
    parser.add_argument('--pitch', type=float, default=25.0, help='camera tilt (deg) when cam2world is unknown')
    parser.add_argument('--cam-height', type=float, default=1.5, help='camera height (m) when cam2world is unknown')
    parser.add_argument('--fov-only', action='store_true', help='only keep voxels visible in the image')
    parser.add_argument('--out', default='outputs')
    args = parser.parse_args()

    camera = {}
    cam_file = args.camera or osp.splitext(args.image)[0] + '.json'
    if osp.exists(cam_file):
        with open(cam_file) as f:
            camera = json.load(f)
        print(f'Using camera parameters from {cam_file}: {sorted(camera)}')
    else:
        print('No camera file found: using default intrinsics and camera pose.')

    model = GPOcc.from_pretrained(args.model, subfolder=args.variant).cuda().eval()
    out = model.predict(
        args.image,
        intrinsics=camera.get('intrinsics'),
        cam2world=camera.get('cam2world'),
        vox_origin=camera.get('vox_origin'),
        pitch_deg=args.pitch,
        cam_height=args.cam_height,
    )

    occ = out['occupancy'].cpu().numpy()                  # (60, 60, 36), 12 = empty
    centers = out['voxel_centers'].cpu().numpy()          # (60, 60, 36, 3), world frame
    keep = (occ != EMPTY_IDX) & (occ != 0)
    if args.fov_only:
        keep &= out['fov_mask'].cpu().numpy()

    os.makedirs(args.out, exist_ok=True)
    stem = osp.splitext(osp.basename(args.image))[0]
    ply_path = osp.join(args.out, f'{stem}_occ.ply')
    save_ply(ply_path, centers[keep], PALETTE[occ[keep]])

    print(f'Saved {keep.sum()} occupied voxels to {ply_path}')
    for c in range(1, EMPTY_IDX):
        n = int((occ[keep] == c).sum())
        if n:
            print(f'  {CLASS_NAMES[c]:<10s} {n}')


if __name__ == '__main__':
    torch.backends.cudnn.benchmark = True
    main()
