# DPA — Diverse Parameters Augmentation Attack on Face Recognition

> **PyTorch implementation of:**
> Zhou et al., *"Improving the Transferability of Adversarial Attacks on Face Recognition with Diverse Parameters Augmentation"*, **CVPR 2025**

No official code was released with the paper. This repository implements the full pipeline from scratch based on the paper's algorithms and mathematical formulations.

---

## What This Paper Does

Face Recognition (FR) models are vulnerable to **adversarial examples** — images with tiny, invisible pixel changes that cause the model to misidentify one person as another. The challenge is **transferability**: an attack crafted against one model often fails on a different model.

DPA solves this by building a diverse collection of surrogate models through two training trajectories (pretrained and random initialization), saving checkpoints at √c-interval epochs, and attacking all of them simultaneously with **Hard Model Aggregation (HMA)** — a technique that injects beneficial perturbations into feature maps to create harder surrogate models, forcing the attack to find more generalizable pixel changes.

```
CASIA-WebFace ──► DPO (train 2 trajectories) ──► ~12 diverse surrogates
                                                          │
LFW pairs ──────────────────────────────────────► HMA attack (200 iters)
                                                          │
                                                   x_adv (adversarial face)
                                                          │
                                              14 victim models ──► ASR table
```

---

## Results (from paper, MobileFace surrogate, LFW dataset)

| Attack | IR152 | IRSE50 | FaceNet | ASR_adv |
|--------|-------|--------|---------|---------|
| DI | 18.4% | 97.3% | 32.9% | ~13% |
| BPFA | 15.8% | 95.5% | 17.8% | ~7% |
| BSR | 5.4% | 74.2% | 9.5% | ~4% |
| **DPA (ours)** | **67.7%** | **98.2%** | **90.8%** | **59.0%** |

---

## Repository Structure

```
DPA/
│
├── run_dpa.py              # Main entry point — full pipeline CLI
├── dpo.py                  # Stage 1: Diverse Parameters Optimization
├── hma.py                  # Stage 2: Hard Model Aggregation attack
├── evaluation.py           # ASR computation at FAR@0.001
│
├── models/
│   ├── backbones.py        # IR152, IRSE50, IR_50, MobileFaceNet architectures
│   └── heads.py            # ArcFace head with additive angular margin loss
│
├── data/
│   └── datasets.py         # CASIA-WebFace, LFW, CelebA-HQ data loaders
│                           # Auto-detects Layout A / B / RecordIO formats
│
├── convert_rec_v4.py       # Convert InsightFace .rec format to image folders
│
├── requirements.txt        # Python dependencies
└── README.md
```

---

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/YOUR_USERNAME/DPA.git
cd DPA
```

**2. Create conda environment**
```bash
conda create -n dpa python=3.10
conda activate dpa
```

**3. Install dependencies**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

`requirements.txt`:
```
torch>=2.1.0
torchvision>=0.16.0
numpy>=1.23.0
Pillow>=9.0.0
tqdm>=4.64.0
scipy>=1.9.0
scikit-learn>=1.3.0
opencv-python>=4.6.0
```

---

## Dataset Setup

### Training Dataset — CASIA-WebFace (for DPO Stage)

CASIA-WebFace contains ~490,000 face images across 10,575 identities.

**Download from Kaggle:**
```bash
pip install kaggle
kaggle datasets download debarghamitraroy/casia-webface
unzip casia-webface.zip -d datasets/CASIA-raw/
```

The downloaded file is in InsightFace RecordIO format (`.rec` + `.idx` + `.lst`). Convert it to image folders:

```bash
# Quick test (1000 images)
python3 convert_rec_v4.py \
    --rec  datasets/CASIA-raw/casia-webface/train.rec \
    --idx  datasets/CASIA-raw/casia-webface/train.idx \
    --lst  datasets/CASIA-raw/casia-webface/train.lst \
    --out  datasets/CASIA-WebFace \
    --limit 1000

# Full conversion (~20 minutes)
python3 convert_rec_v4.py \
    --rec  datasets/CASIA-raw/casia-webface/train.rec \
    --idx  datasets/CASIA-raw/casia-webface/train.idx \
    --lst  datasets/CASIA-raw/casia-webface/train.lst \
    --out  datasets/CASIA-WebFace \
    --limit 0
```

Expected result:
```
Identities : 10,572
Images     : 490,623
```

> The original paper uses BUPT-Balancedface (available via email request at http://www.whdeng.cn/RFW/). CASIA-WebFace is a standard publicly available substitute.

---

### Attack & Evaluation Dataset — LFW

LFW contains 13,233 face images across 5,749 identities.

```bash
# Download images
wget http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz
tar -xzf lfw-deepfunneled.tgz
mv lfw-deepfunneled datasets/LFW

# Download pairs.txt (if UMass server is unreachable use Figshare mirror)
wget http://vis-www.cs.umass.edu/lfw/pairs.txt -O datasets/LFW/pairs.txt

# Figshare mirror (if above fails)
python3 -c "
import urllib.request
urllib.request.urlretrieve(
    'https://ndownloader.figshare.com/files/5976006',
    'datasets/LFW/pairs.txt'
)
print('Done')
"
```

Expected result:
```
Identities : 5,749
Images     : 13,233
```

---

### Pretrained Surrogate Weights (optional but recommended)

Without pretrained weights, both DPO trajectories start from random initialization. Providing pretrained weights gives Trajectory P a head start and improves final ASR.

```bash
mkdir -p pretrained/

# MobileFace (paper's surrogate)
wget https://github.com/foamliu/MobileFaceNet/releases/download/v1.0/mobilefacenet.pt \
     -O pretrained/mobilefacenet.pth

# IR_50 (alternative surrogate, more stable)
wget https://github.com/deepinsight/insightface/releases/download/v0.7/ms1mv3_arcface_r50_fp16.pth \
     -O pretrained/IR_50.pth
```

---

### Victim Model Weights

Victim models are only needed for evaluation (Stage 3). They are never used during attack crafting.

| Model | Source |
|-------|--------|
| IR152, IRSE50, MobileFace | BPFA paper repo: https://github.com/zhFuECL/BPFA |
| CurricularFace, MagFace, ArcFace, CircleLoss, MV-Softmax, NPCFace | FaceX-ZOO: https://github.com/JDAI-CV/FaceX-Zoo |
| FaceNet | `pip install facenet-pytorch` |

```bash
# FaceNet (easiest — downloads automatically)
pip install facenet-pytorch
python3 -c "
import torch
from facenet_pytorch import InceptionResnetV1
m = InceptionResnetV1(pretrained='vggface2').eval()
torch.save(m.state_dict(), 'pretrained/victims/FaceNet.pth')
print('Saved')
"
```

Place all victim model `.pth` files in `pretrained/victims/`.

---

## Running the Pipeline

### Quick Test (~5 minutes, verifies pipeline works)

```bash
python3 run_dpa.py \
    --train_dir  datasets/CASIA-WebFace \
    --attack_dir datasets/LFW \
    --surrogate  IR_50 \
    --max_ids    100 \
    --epochs     2 \
    --n_iters    5 \
    --num_pairs  3 \
    --output_dir ./outputs_test
```

### Full Run — Paper Settings (~12 hours on server GPU)

```bash
python3 run_dpa.py \
    --train_dir  datasets/CASIA-WebFace \
    --attack_dir datasets/LFW \
    --pretrained pretrained/mobilefacenet.pth \
    --victim_dir pretrained/victims/ \
    --surrogate  MobileFace \
    --epochs     20 \
    --n_iters    200 \
    --num_pairs  200 \
    --output_dir ./outputs_full
```

### Skip DPO — Load Saved Checkpoints

```bash
python3 run_dpa.py \
    --load_ckpts ./outputs_full/checkpoints \
    --attack_dir datasets/LFW \
    --victim_dir pretrained/victims/ \
    --surrogate  MobileFace \
    --n_iters    200 \
    --num_pairs  200 \
    --output_dir ./outputs_attack_only
```

---

## All Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--train_dir` | required | Path to training dataset (CASIA-WebFace) |
| `--attack_dir` | required | Path to attack dataset (LFW) |
| `--pretrained` | None | Path to pretrained backbone `.pth` |
| `--victim_dir` | None | Folder of victim model `.pth` files |
| `--load_ckpts` | None | Load saved DPO checkpoints (skips DPO) |
| `--surrogate` | MobileFace | Backbone: `MobileFace` / `IR_50` / `IR_152` / `IRSE_50` |
| `--emb_size` | 512 | Embedding dimension |
| `--img_size` | 112 | Input image size |
| `--epochs` | 20 | DPO training epochs per trajectory |
| `--lr` | 0.1 | Initial learning rate |
| `--batch_size` | 64 | Training batch size |
| `--margin` | 0.5 | ArcFace angular margin m |
| `--scale` | 64.0 | ArcFace scale factor d |
| `--n_iters` | 200 | HMA attack iterations |
| `--eps` | 10/255 | L∞ perturbation budget ε |
| `--beta` | 1/255 | Adversarial step size β |
| `--eta` | 1/255 | Beneficial perturbation step size η |
| `--layer_frac` | 0.3 | Fraction of layers for perturbation |
| `--num_pairs` | 200 | Number of source-target pairs to attack |
| `--max_ids` | None | Limit identities for quick tests |
| `--output_dir` | ./outputs | Output directory |
| `--device` | auto | `cuda` or `cpu` |

---

## How It Works

### Stage 1 — Diverse Parameters Optimization (DPO)

DPO builds a diverse set of surrogate models by training from two different initializations:

- **Trajectory P**: backbone starts from pretrained weights, head starts random
- **Trajectory A**: both backbone and head start from random (Kaiming) initialization

Each trajectory trains for `c` epochs using ArcFace loss. Checkpoints are saved at interval κ = ⌊√c⌋. The union of both trajectories' checkpoints forms **Vᶜq** — approximately 12 diverse surrogate models.

```
Pretrained ──► train 20 epochs ──► save at epochs 4,8,12,16,20  ─┐
                                                                   ├──► Vᶜq (~12 models)
Random     ──► train 20 epochs ──► save at epochs 4,8,12,16,20  ─┘
```

### Stage 2 — Hard Model Aggregation (HMA)

HMA crafts adversarial examples by:

1. Loading all ~12 surrogate models from Vᶜq
2. For each of 200 iterations:
   - Apply random resize+pad transform to x_adv
   - Inject beneficial perturbations ω into selected feature maps (makes models harder)
   - Compute average embedding-distance loss across all 12 models
   - Update x_adv with sign-SGD step: `x_adv -= β × sign(∇loss)`
   - Clip to ε-ball around original source image

The beneficial perturbations push feature maps in the direction that increases the loss — simulating harder model variants. If x_adv can fool these hardened models, it generalizes to unseen victim models.

### Stage 3 — Evaluation

For each victim model:
1. Compute FAR@0.001 threshold on full LFW pairs
2. Feed x_adv and x_target through victim model
3. Compute cosine similarity between embeddings
4. If similarity ≥ threshold → attack succeeded
5. ASR = fraction of successful attacks over all pairs

---

## Output Files

```
outputs/
├── checkpoints/
│   ├── surrogate_000.pth     ← v₀ᵖ (initial pretrained weights)
│   ├── surrogate_001.pth     ← epoch 4, trajectory P
│   ├── surrogate_002.pth     ← epoch 8, trajectory P
│   ├── ...
│   └── surrogate_011.pth     ← epoch 20, trajectory A
│
├── adv_examples.pt           ← dict with 'adv' and 'targets' lists
└── results.json              ← ASR per victim model
```

`results.json` format:
```json
{
  "IR152":      0.677,
  "IRSE50":     0.982,
  "FaceNet":    0.908,
  "MobileFace": 0.716
}
```

---

## Hardware Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| GPU VRAM | 8 GB | 24 GB |
| System RAM | 16 GB | 32 GB+ |
| Storage | 25 GB | 50 GB |
| DPO runtime | ~45 min/epoch | ~10 min/epoch |
| HMA runtime | ~5 min/pair | ~1 min/pair |
| Total (20 epochs, 200 pairs) | ~46 hrs | ~12 hrs |

> For configurations exceeding 16 GB VRAM, enable mixed precision:
> add `--device cuda` and use `torch.cuda.amp.autocast()` (built into the training loop).

---

## Troubleshooting

**`RuntimeError: element 0 of tensors does not require grad`**
The computation graph from x_adv to the loss is broken. Make sure `x_adv.requires_grad_(True)` is set before the forward pass in `hma.py`.

**`ValueError: Expected more than 1 value per channel when training`**
BatchNorm1d receives batch size 1 in train mode. Set models to `.eval()` mode during the attack.

**Distance between embeddings is 0.0**
The backbone is outputting identical vectors for all inputs. This happens with MobileFaceNet when BatchNorm running stats are uninitialised (fresh random weights + eval mode). Use IR_50 as the surrogate instead, or train the surrogate first before attacking.

**`FileNotFoundError: No such file or directory: BUPT-Balancedface`**
Use CASIA-WebFace instead — it is a standard substitute. See Dataset Setup above.

**`[Victims] No victim_dir provided — using random models for demo`**
ASR of 100% with random victim models is meaningless. Download real pretrained victim models and pass `--victim_dir`.

---

## Citation

If you use this code, please cite the original paper:

```bibtex
@inproceedings{zhou2025dpa,
  title     = {Improving the Transferability of Adversarial Attacks on
               Face Recognition with Diverse Parameters Augmentation},
  author    = {Zhou, Fengfan and Yin, Bangjie and Ling, Hefei and
               Zhou, Qianyu and Wang, Wenxuan},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer
               Vision and Pattern Recognition (CVPR)},
  pages     = {3516--3527},
  year      = {2025}
}
```

---

## References

- [1] Zhou et al., DPA, CVPR 2025
- [2] Deng et al., ArcFace, TPAMI 2022
- [3] Gubri et al., LGV, ECCV 2022
- [4] Xie et al., DI, CVPR 2019
- [5] Zhou et al., BPFA, IEEE TCSS 2024
- [6] Wang et al., FaceX-ZOO, 2021
