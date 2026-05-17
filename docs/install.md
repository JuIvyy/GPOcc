# Installation

## Prerequisites

- Python 3.11
- CUDA 12.1
- GCC >= 7.5

## 1. Clone the Repository

```bash
git clone https://github.com/xxx/GPOcc.git
cd GPOcc
```

## 2. Create a Conda Environment

```bash
conda create -n gpocc python=3.11 -y
conda activate gpocc
```

## 3. Install PyTorch

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu121
```

## 4. Install MM-Series Libraries

```bash
pip install openmim
mim install mmcv==2.1.0
mim install mmdet==3.2.0
mim install mmsegmentation==1.2.2
mim install mmdet3d==1.4.0
```

## 6. Install Other Packages

```bash
conda install pytorch3d -c pytorch3d

pip install tqdm einops open3d timm
```

Install [pytorch_cluster](https://github.com/rusty1s/pytorch_cluster) and [pytorch_scatter](https://github.com/rusty1s/pytorch_scatter)


## 5. Build Custom CUDA Operators

```bash
cd src/gpocc/model/head/gaussian_occ_head/ops/localagg_prob_gf2
python setup.py build_ext --inplace
cd -
```

## 6. Download Pretrained Checkpoints

**VGGT-1B** — downloaded from [HuggingFace](https://huggingface.co/facebook/VGGT-1B)

**Depth-Anything-V2 (finetuned on OccScanNet)** — download from [YkiWu/EmbodiedOcc](https://huggingface.co/YkiWu/EmbodiedOcc) 

Download to ${HF_HOME}/hub