# Training and Evaluation

All scripts should be run from the `scripts/` directory.


## Training

Train GPOcc on OccScanNet with the monocular single-frame setting:

```bash
bash scripts/train.sh <NUM_GPUS> <CONFIG>
```

**Example** — 4-GPU training with the default config:

```bash
bash scripts/train.sh 4 mono_dpt_bin16
```

This runs `train_mono.py` with:
- Config: `config/<CONFIG>.py`
- Output: `work_dirs/<CONFIG>/`


---

## Evaluation

### Monocular (single-frame) evaluation

Run evaluation on the OccScanNet test set:

```bash
bash train.sh <NUM_GPUS> <CONFIG> --evaluate
```

**Example:**

```bash
bash train.sh 4 mono_dpt_bin16_release --evaluate
```

The checkpoint is loaded from `work_dirs/<CONFIG>/latest.pth` by default. To load a specific checkpoint, set `load_from` in the config.

---

### Embodied (streaming / incremental fusion) evaluation

Evaluate the model in the multi-frame streaming setting on EmbodiedOcc-ScanNet:

```bash
bash eval_embodied_for_mono.sh <CONFIG>
```

**Example:**

```bash
bash eval_embodied_for_mono.sh mono_dpt_bin16_release

# or 

bash scripts/eval_embodied_for_mono_vggt.sh mono_vggt_bin16_release
```
