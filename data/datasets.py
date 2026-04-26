"""
Dataset utilities for DPA.

Supports:
  - FRTrainDataset  : generic folder-based dataset for DPO training
  - FRPairDataset   : source-target pair dataset for attack evaluation
  - LFWPairDataset  : standard LFW pairs.txt evaluation protocol
  - CelebAHQDataset : CelebA-HQ identity pairs for attack
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Optional, Callable

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Standard face preprocessing transforms
# ──────────────────────────────────────────────────────────────────────────────

def get_train_transform(img_size: int = 112) -> transforms.Compose:
    """
    Data augmentation used during DPO training.
    Matches standard FR training pipelines.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])


def get_eval_transform(img_size: int = 112) -> transforms.Compose:
    """
    Deterministic preprocessing for attack / evaluation.
    """
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """
    Convert from normalized [-1, 1] range back to [0, 1].
    mean=0.5, std=0.5  →  x_orig = x * 0.5 + 0.5
    """
    return x * 0.5 + 0.5


def normalize_for_model(x: torch.Tensor) -> torch.Tensor:
    """Convert [0, 1] image tensor to [-1, 1] for model input."""
    return (x - 0.5) / 0.5


# ──────────────────────────────────────────────────────────────────────────────
# Training dataset (for DPO)
# ──────────────────────────────────────────────────────────────────────────────

class FRTrainDataset(Dataset):
    """
    Generic face recognition training dataset.
    Expected directory structure:
        root/
          identity_0001/
              img1.jpg
              img2.jpg
          identity_0002/
              ...

    Parameters
    ----------
    root      : path to root directory
    transform : torchvision transform
    min_imgs  : minimum images per identity (filters sparse identities)
    """

    def __init__(self,
             root: str,
             transform: Optional[Callable] = None,
             min_imgs: int = 2,
             img_size: int = 112,
             max_ids: int = None,
             max_imgs_per_id: int = 20):
        self.transform = transform or get_train_transform(img_size)
        self.samples: List[Tuple[Path, int]] = []
        self.root      = Path(root)
        self.class_to_idx: dict = {}
        identity_dirs = sorted([d for d in self.root.iterdir()

                                  if d.is_dir()])

        # Limit number of identities

        if max_ids:
            identity_dirs = identity_dirs[:max_ids]

        class_idx = 0
        for id_dir in identity_dirs:
            imgs = list(id_dir.glob('*.jpg')) + \
               list(id_dir.glob('*.png')) + \
               list(id_dir.glob('*.jpeg'))

        # Skip identities with too few images
            if len(imgs) < min_imgs:
                continue

        # Limit images per identity
            if max_imgs_per_id:
                imgs = imgs[:max_imgs_per_id]

            self.class_to_idx[id_dir.name] = class_idx
            for img_path in imgs:
                self.samples.append((img_path, class_idx))
            class_idx += 1

        self.num_classes = class_idx
        print(f"[Dataset] {self.root.name}: "
              f"{len(self.samples)} images, {self.num_classes} identities")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label


# ──────────────────────────────────────────────────────────────────────────────
# Attack pair dataset (source + target images)
# ──────────────────────────────────────────────────────────────────────────────

class FRPairDataset(Dataset):
    """
    Dataset of (source_image, target_image) pairs for adversarial attack.

    Given a root directory with identity folders, randomly pairs
    source and target images from *different* identities.

    Parameters
    ----------
    root        : directory with identity subfolders
    num_pairs   : how many source-target pairs to generate
    img_size    : resize dimension
    seed        : for reproducibility
    """

    def __init__(self,
                 root: str,
                 num_pairs: int = 200,
                 img_size:  int = 112,
                 seed:      int = 42):
        self.transform = get_eval_transform(img_size)
        rng = random.Random(seed)

        root_path = Path(root)
        identity_dirs = sorted([d for d in root_path.iterdir() if d.is_dir()])

        # Collect all images per identity
        id_to_imgs: dict = {}
        for id_dir in identity_dirs:
            imgs = list(id_dir.glob('*.jpg')) + \
                   list(id_dir.glob('*.png')) + \
                   list(id_dir.glob('*.jpeg'))
            if imgs:
                id_to_imgs[id_dir.name] = imgs

        ids = list(id_to_imgs.keys())
        assert len(ids) >= 2, "Need at least 2 identities for pair generation"

        self.pairs: List[Tuple[Path, Path]] = []
        for _ in range(num_pairs):
            src_id, tgt_id = rng.sample(ids, 2)
            src_img = rng.choice(id_to_imgs[src_id])
            tgt_img = rng.choice(id_to_imgs[tgt_id])
            self.pairs.append((src_img, tgt_img))

        print(f"[PairDataset] Generated {len(self.pairs)} source-target pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, tgt_path = self.pairs[idx]
        src = Image.open(src_path).convert('RGB')
        tgt = Image.open(tgt_path).convert('RGB')
        return self.transform(src), self.transform(tgt)


# ──────────────────────────────────────────────────────────────────────────────
# LFW pairs evaluation dataset
# ──────────────────────────────────────────────────────────────────────────────

class LFWPairDataset(Dataset):
    """
    Standard LFW pairs evaluation using pairs.txt.

    pairs.txt format:
      Line 1: num_folds  num_pairs_per_fold
      Same-person pairs: name  img1_idx  img2_idx
      Diff-person pairs: name1 img1_idx  name2  img2_idx

    The dataset returns (img1, img2, is_same) tuples.
    """

    def __init__(self,
                 lfw_dir:    str,
                 pairs_file: str,
                 img_size:   int = 112):
        self.transform = get_eval_transform(img_size)
        self.lfw_dir   = Path(lfw_dir)

        self.pairs: List[Tuple[Path, Path, int]] = []
        self._parse_pairs(pairs_file)

        print(f"[LFW] Loaded {len(self.pairs)} pairs")

    def _img_path(self, name: str, idx: int) -> Path:
        fname = f"{name}_{idx:04d}.jpg"
        return self.lfw_dir / name / fname

    def _parse_pairs(self, pairs_file: str):
        with open(pairs_file, 'r') as f:
            lines = f.read().strip().split('\n')

        # First line: num_folds pairs_per_fold
        meta = lines[0].split()
        # Rest: pair definitions
        for line in lines[1:]:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) == 3:
                # Same-person pair
                name, idx1, idx2 = parts[0], int(parts[1]), int(parts[2])
                p1 = self._img_path(name, idx1)
                p2 = self._img_path(name, idx2)
                self.pairs.append((p1, p2, 1))
            elif len(parts) == 4:
                # Different-person pair
                name1, idx1, name2, idx2 = (parts[0], int(parts[1]),
                                             parts[2], int(parts[3]))
                p1 = self._img_path(name1, idx1)
                p2 = self._img_path(name2, idx2)
                self.pairs.append((p1, p2, 0))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p1, p2, is_same = self.pairs[idx]
        img1 = Image.open(p1).convert('RGB')
        img2 = Image.open(p2).convert('RGB')
        return self.transform(img1), self.transform(img2), is_same


# ──────────────────────────────────────────────────────────────────────────────
# CelebA-HQ attack pair dataset
# ──────────────────────────────────────────────────────────────────────────────

class CelebAHQDataset(Dataset):
    """
    CelebA-HQ pairs for adversarial attack.
    Assumes identity labels are available via a text file:
        img_path  identity_label
    """

    def __init__(self,
                 img_dir:      str,
                 identity_file: str,
                 num_pairs:    int = 200,
                 img_size:     int = 112,
                 seed:         int = 42):
        self.transform = get_eval_transform(img_size)
        rng = random.Random(seed)

        img_dir_path = Path(img_dir)
        id_to_imgs: dict = {}

        with open(identity_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
                img_name, identity = parts[0], parts[1]
                img_path = img_dir_path / img_name
                if img_path.exists():
                    if identity not in id_to_imgs:
                        id_to_imgs[identity] = []
                    id_to_imgs[identity].append(img_path)

        ids = list(id_to_imgs.keys())
        self.pairs: List[Tuple[Path, Path]] = []
        for _ in range(num_pairs):
            src_id, tgt_id = rng.sample(ids, 2)
            src_img = rng.choice(id_to_imgs[src_id])
            tgt_img = rng.choice(id_to_imgs[tgt_id])
            self.pairs.append((src_img, tgt_img))

        print(f"[CelebAHQ] Generated {len(self.pairs)} source-target pairs")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        src_path, tgt_path = self.pairs[idx]
        src = Image.open(src_path).convert('RGB')
        tgt = Image.open(tgt_path).convert('RGB')
        return self.transform(src), self.transform(tgt)


# ──────────────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────────────

def get_train_loader(root, batch_size=32, num_workers=0,
                     img_size=112, max_ids=None,
                     max_imgs_per_id=50):
    dataset = FRTrainDataset(
        root=root,
        img_size=img_size,
        max_ids=max_ids,
        max_imgs_per_id=max_imgs_per_id
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True
    )
    return loader, dataset.num_classes

def get_attack_loader(root:       str,
                      num_pairs:  int = 200,
                      img_size:   int = 112) -> DataLoader:
    """Returns dataloader of (source, target) pairs for attack."""
    ds = FRPairDataset(root=root, num_pairs=num_pairs, img_size=img_size)
    return DataLoader(ds, batch_size=1, shuffle=False, num_workers=2)