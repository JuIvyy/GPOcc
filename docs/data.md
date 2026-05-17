# Data Preparation

GPOcc is trained and evaluated on two datasets: **OccScanNet** and **EmbodiedOcc-ScanNet**.

## Directory Structure

After preparation, the `data/` directory should look like:

```
data/
├── occscannet/
│   ├── train_final.txt
│   ├── test_final.txt
│   ├── train_mini_final.txt
│   ├── test_mini_final.txt
│   ├── gathered_data -> /path/to/OccScanNet/gathered_data
│   └── posed_images  -> /path/to/OccScanNet/posed_images
└── scene_occ/
    ├── train_online.txt
    ├── test_online.txt
    ├── train_mini_online.txt
    ├── test_mini_online.txt
    ├── global_occ_package       -> /path/to/EmbodiedOcc-ScanNet/global_occ_package
    └── streme_occ_new_package   -> /path/to/EmbodiedOcc-ScanNet/streme_occ_new_package
```

## OccScanNet

1. Download the dataset from [hongxiaoy/OccScanNet](https://huggingface.co/datasets/hongxiaoy/OccScanNet).

2. Unzip the downloaded files.

3. Create symbolic links under `data/occscannet/`:

```bash
cd data/occscannet
ln -s /path/to/OccScanNet/gathered_data
ln -s /path/to/OccScanNet/posed_images
cd ../..
```

## EmbodiedOcc-ScanNet

Used for the embodied (streaming) evaluation setting.

1. Download the dataset from [YkiWu/EmbodiedOcc-ScanNet](https://huggingface.co/datasets/YkiWu/EmbodiedOcc-ScanNet).

2. Unzip the downloaded files.

3. Create symbolic links under `data/scene_occ/`:

```bash
cd data/scene_occ
ln -s /path/to/EmbodiedOcc-ScanNet/global_occ_package
ln -s /path/to/EmbodiedOcc-ScanNet/streme_occ_new_package
cd ../..
```

## Mini Sets

The split files `*_mini_*.txt` provide a small subset for quick validation runs. Set `data_tg='mini'` in the dataset config to use them.
