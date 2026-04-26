"""
Stage 1 — Diverse Parameters Optimization (DPO)
================================================
Implements Algorithm 1 from the DPA paper exactly.

Key idea:
  Two initializations → two training trajectories → checkpoints saved
  at interval κ = ⌊√c⌋ → union forms Vᶜq

Initialization sets:
  P = {v₀ᵖ, w₀ᵖ}  — pre-trained backbone, random head
  A = {v₀ᵃ, w₀ᵃ}  — fully random backbone + head

Output: Vᶜq  (list of backbone state_dicts)
"""

import os
import math
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import List, Dict, Optional

from models.backbones import get_model
from models.heads import ArcFaceHead


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint interval rule  κ = ⌊√c⌋
# ──────────────────────────────────────────────────────────────────────────────

def get_save_epochs(num_epochs: int) -> List[int]:
    """
    Returns the list of epoch indices (1-indexed) at which to save checkpoints.
    Rule from Equation (11): j mod κ = 1, κ = ⌊√c⌋

    Example: c=25 → κ=5 → save at epochs 1,6,11,16,21,25
    """
    kappa = max(1, int(math.floor(math.sqrt(num_epochs))))
    epochs = [i for i in range(1, num_epochs + 1) if (i % kappa == 1)]
    if num_epochs not in epochs:
        epochs.append(num_epochs)          # always save final epoch
    return sorted(set(epochs))


# ──────────────────────────────────────────────────────────────────────────────
# Single training run
# ──────────────────────────────────────────────────────────────────────────────

def train_one_trajectory(
    backbone: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    num_epochs: int,
    lr: float,
    device: torch.device,
    save_epochs: List[int],
    desc: str = "Training"
) -> List[Dict]:
    """
    Trains backbone + head for num_epochs using ArcFace loss.
    Saves and returns backbone state_dicts at save_epochs.

    Returns
    -------
    checkpoints : List of backbone state_dicts (deep copies, on CPU)
    """
    backbone = backbone.to(device)
    head     = head.to(device)

    optimizer = optim.SGD(
        list(backbone.parameters()) + list(head.parameters()),
        lr=lr, momentum=0.9, weight_decay=5e-4
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    checkpoints = []
    save_epoch_set = set(save_epochs)

    for epoch in range(1, num_epochs + 1):
        backbone.train()
        head.train()
        total_loss = 0.0
        n_batches  = 0

        pbar = tqdm(dataloader, desc=f"{desc} epoch {epoch}/{num_epochs}",
                    leave=False)
        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            embeddings = backbone(images)
            loss = head(embeddings, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        print(f"  [{desc}] Epoch {epoch}/{num_epochs} — "
              f"avg loss: {avg_loss:.4f}  lr: {scheduler.get_last_lr()[0]:.5f}")

        # ── Save checkpoint if this epoch is in the save set ──────────
        if epoch in save_epoch_set:
            ckpt = copy.deepcopy(backbone).cpu().state_dict()
            checkpoints.append(ckpt)
            print(f"    ✓ Saved checkpoint at epoch {epoch} "
                  f"(total saved: {len(checkpoints)})")

    return checkpoints


# ──────────────────────────────────────────────────────────────────────────────
# DPO — main function (Algorithm 1)
# ──────────────────────────────────────────────────────────────────────────────

def run_dpo(
    model_name:      str,
    num_classes:     int,
    dataloader:      DataLoader,
    num_epochs:      int       = 20,
    lr:              float     = 0.1,
    emb_size:        int       = 512,
    pretrained_path: Optional[str] = None,
    device:          Optional[torch.device] = None,
    save_dir:        Optional[str] = None,
    margin:          float     = 0.5,
    scale:           float     = 64.0,
    input_size:      tuple     = (112, 112),
) -> List[Dict]:
    """
    Full DPO stage — Algorithm 1 of DPA paper.

    Steps
    -----
    1. Build P = {v₀ᵖ, w₀ᵖ}  (pretrained backbone, random head)
    2. Build A = {v₀ᵃ, w₀ᵃ}  (random backbone, random head)
    3. For each init set J ∈ {P, A}:
         a. Map parameters to model
         b. Train for c epochs with ArcFace loss
         c. Save backbone at κ-interval epochs → add to Vᶜq
    4. Return Vᶜq

    Parameters
    ----------
    model_name      : backbone architecture, e.g. 'MobileFace'
    num_classes     : number of identities in training dataset
    dataloader      : DataLoader yielding (images, labels)
    num_epochs      : c in the paper
    lr              : initial learning rate
    emb_size        : embedding dimension
    pretrained_path : path to pre-trained backbone weights (.pth)
                      If None, both trajectories start from random init
    device          : torch.device (auto-detected if None)
    save_dir        : directory to save checkpoint .pth files (optional)
    margin          : ArcFace angular margin m
    scale           : ArcFace scale factor d
    input_size      : (H, W) of input face crops

    Returns
    -------
    Vq_c : List of backbone state_dicts (CPU tensors)
           First element is always v₀ᵖ (pretrained / init of P trajectory)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[DPO] Device: {device}")

    # Checkpoint interval κ = ⌊√c⌋
    save_epochs = get_save_epochs(num_epochs)
    print(f"[DPO] c={num_epochs}, κ={int(math.floor(math.sqrt(num_epochs)))}, "
          f"save epochs: {save_epochs}")

    Vq_c = []   # Will hold all collected backbone state_dicts

    # ── Initialization set P: pretrained backbone + random head ───────────
    print("\n[DPO] ── Trajectory P (pre-trained init) ──────────────────────")
    backbone_P = get_model(model_name, input_size=input_size, emb_size=emb_size)

    if pretrained_path and os.path.isfile(pretrained_path):
        state = torch.load(pretrained_path, map_location='cpu')
        # Handle wrapped state_dicts
        if 'state_dict' in state:
            state = state['state_dict']
        
        # Filter out bn layer parameters with shape mismatches (emb_size mismatch)
        model_state = backbone_P.state_dict()
        filtered_state = {}
        for key, val in state.items():
            if key in model_state:
                if val.shape == model_state[key].shape:
                    filtered_state[key] = val
                else:
                    print(f"  ⚠ Skipping {key}: shape mismatch {val.shape} vs {model_state[key].shape}")
            else:
                filtered_state[key] = val
        
        backbone_P.load_state_dict(filtered_state, strict=False)
        print(f"  Loaded pre-trained weights from: {pretrained_path}")
        # ── ADD THIS VERIFICATION BLOCK HERE ──────────────────────────
        model_keys = set(model_state.keys())
        ckpt_keys  = set(state.keys())
        matched    = set(filtered_state.keys())
        print(f"  [Verify] Total model keys   : {len(model_keys)}")
        print(f"  [Verify] Total ckpt keys    : {len(ckpt_keys)}")
        print(f"  [Verify] Matched keys       : {len(matched)}")
        print(f"  [Verify] Missing from ckpt  : {len(model_keys - ckpt_keys)}")
        skipped = {k for k, v in state.items() 
                   if k in model_state and v.shape != model_state[k].shape}
        print(f"  [Verify] Skipped (mismatch) : {len(skipped)}")
        if len(model_keys - ckpt_keys) == 0 and len(skipped) == 0:
            print(f"  [Verify] ✓ All weights loaded perfectly")
        else:
            print(f"  [Verify] ✗ Some weights missing — check MobileFaceNet architecture")
        # ── END VERIFICATION BLOCK ────────────────────────────────────
    else:
        print("  No pretrained path provided — P trajectory starts from random init")

    # Always save v₀ᵖ as the first checkpoint (Equation 11)
    Vq_c.append(copy.deepcopy(backbone_P).cpu().state_dict())
    print(f"  ✓ Saved v₀ᵖ (epoch-0 pretrained checkpoint)")

    head_P = ArcFaceHead(emb_size=emb_size, num_classes=num_classes,
                         margin=margin, scale=scale)

    ckpts_P = train_one_trajectory(
        backbone    = backbone_P,
        head        = head_P,
        dataloader  = dataloader,
        num_epochs  = num_epochs,
        lr          = lr,
        device      = device,
        save_epochs = save_epochs,
        desc        = "Trajectory-P"
    )
    Vq_c.extend(ckpts_P)
    print(f"  Trajectory P produced {len(ckpts_P)} checkpoints")

    # ── Initialization set A: fully random backbone + random head ─────────
    print("\n[DPO] ── Trajectory A (random init) ───────────────────────────")
    backbone_A = get_model(model_name, input_size=input_size, emb_size=emb_size)
    # backbone_A already has Kaiming-normal random weights from __init__
    head_A = ArcFaceHead(emb_size=emb_size, num_classes=num_classes,
                         margin=margin, scale=scale)

    ckpts_A = train_one_trajectory(
        backbone    = backbone_A,
        head        = head_A,
        dataloader  = dataloader,
        num_epochs  = num_epochs,
        lr          = lr,
        device      = device,
        save_epochs = save_epochs,
        desc        = "Trajectory-A"
    )
    Vq_c.extend(ckpts_A)
    print(f"  Trajectory A produced {len(ckpts_A)} checkpoints")

    print(f"\n[DPO] Total surrogate models in Vᶜq: {len(Vq_c)}")

    # ── Optionally persist checkpoints to disk ────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        for idx, ckpt in enumerate(Vq_c):
            path = os.path.join(save_dir, f"surrogate_{idx:03d}.pth")
            torch.save(ckpt, path)
        print(f"[DPO] Checkpoints saved to: {save_dir}")

    return Vq_c


# ──────────────────────────────────────────────────────────────────────────────
# Utility: load Vᶜq from disk
# ──────────────────────────────────────────────────────────────────────────────

def load_vq_c(save_dir: str) -> List[Dict]:
    """Load all surrogate_*.pth files from save_dir, sorted by name."""
    import glob
    paths = sorted(glob.glob(os.path.join(save_dir, "surrogate_*.pth")))
    if not paths:
        raise FileNotFoundError(f"No surrogate_*.pth found in {save_dir}")
    checkpoints = [torch.load(p, map_location='cpu') for p in paths]
    print(f"[DPO] Loaded {len(checkpoints)} checkpoints from {save_dir}")
    return checkpoints