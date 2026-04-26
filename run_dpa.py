"""
DPA — Complete End-to-End Pipeline
"""

import os
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional

torch.backends.cudnn.enabled = False

from models.backbones import get_model
from data.datasets import get_train_loader, get_attack_loader
from dpo import run_dpo, load_vq_c
from hma import HMAAttack
from evaluation import DPAEvaluator

from facenet_pytorch import InceptionResnetV1
# ──────────────────────────────────────────────────────────────────────────────
# Argument parser
# ──────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="DPA: Diverse Parameters Augmentation Attack")

    # Data paths
    p.add_argument('--train_dir',   type=str, default=None,
                   help='Root dir of FR training dataset (for DPO)')
    p.add_argument('--attack_dir',  type=str, required=True,
                   help='Root dir of attack dataset (source-target pairs)')
    p.add_argument('--victim_dir',  type=str, default=None,
                   help='Directory containing victim model .pth files')
    p.add_argument('--pretrained',  type=str, default=None,
                   help='Path to pre-trained surrogate backbone weights')
    p.add_argument('--load_ckpts',  type=str, default=None,
                   help='Load saved DPO checkpoints from this dir (skips DPO)')

    # Model settings
    p.add_argument('--surrogate',   type=str, default='MobileFace',
                   choices=['MobileFace', 'IR_50', 'IR_152', 'IRSE_50'],
                   help='Surrogate model architecture')
    p.add_argument('--emb_size',    type=int, default=512,
                   help='Embedding dimension for surrogate and victim models')
    p.add_argument('--img_size',    type=int, default=112)

    # DPO settings
    p.add_argument('--max_ids',     type=int,   default=None,
                   help='Limit number of identities for quick runs')
    p.add_argument('--epochs',      type=int,   default=20,
                   help='Number of training epochs c')
    p.add_argument('--lr',          type=float, default=0.1,
                   help='Initial learning rate for DPO')
    p.add_argument('--batch_size',  type=int,   default=2,
                   help='Batch size (use 2-4 due to CUDA stability)')
    p.add_argument('--num_workers', type=int,   default=0,
                   help='Number of data loading workers (use 0 for stability)')
    p.add_argument('--margin',      type=float, default=0.5,
                   help='ArcFace angular margin m')
    p.add_argument('--scale',       type=float, default=64.0,
                   help='ArcFace scale factor d')

    # HMA / attack settings
    p.add_argument('--n_iters',     type=int,   default=200,
                   help='Number of attack iterations n')
    p.add_argument('--eps',         type=float, default=10.0/255.0,
                   help='L∞ perturbation budget ε')
    p.add_argument('--beta',        type=float, default=1.0/255.0,
                   help='Adversarial step size β')
    p.add_argument('--eta',         type=float, default=1.0/255.0,
                   help='Beneficial perturbation step size η')
    p.add_argument('--layer_frac',  type=float, default=0.3,
                   help='Fraction of layers for beneficial perturbation |Ω|')
    p.add_argument('--num_pairs',   type=int,   default=200,
                   help='Number of source-target pairs to attack')

    # Output
    p.add_argument('--output_dir',  type=str,   default='./outputs')
    p.add_argument('--device',      type=str,   default=None,
                   help='cuda / cpu (auto-detected if not set)')

    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────────
# Default pretrained paths for surrogate models
# ──────────────────────────────────────────────────────────────────────────────

def get_default_pretrained_path(model_name: str) -> Optional[str]:
    """
    Returns the default pretrained path for a given surrogate model.
    Maps model names to their pre-trained weights in ./pretrained/ directory.
    
    Parameters
    ----------
    model_name : str
        Name of the surrogate model (e.g., 'MobileFace', 'IR_50', 'IR_152', 'IRSE_50')
    
    Returns
    -------
    str or None
        Path to pretrained weights, or None if not found
    """
    # Map surrogate models to their default pretrained paths
    default_paths = {
        'MobileFace': './pretrained/MobileFaceNet.pth',
        'IR_50':      './pretrained/victims/IR_50.pth',
        'IR_152':     './pretrained/victims/IR_152.pth',
        'IRSE_50':    './pretrained/victims/IRSE_50.pth',
        # 'facenet':'./pretrained/victims/FaceNet.pth'
    }
    
    path = default_paths.get(model_name)
    if path and os.path.isfile(path):
        return path
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Load victim models
# ──────────────────────────────────────────────────────────────────────────────

def load_victim_models(victim_dir:   str,
                       device:       torch.device,
                       img_size:     int = 112,
                       emb_size:     int = 512) -> Dict[str, torch.nn.Module]:
    """
    Load all victim models from victim_dir.
    Expects files named like:  IR152.pth, IRSE50.pth, FaceNet.pth, etc.

    Falls back to randomly initialised models if files not found
    (for testing/demo purposes).
    """
    if victim_dir is None:
        print("[Victims] No victim_dir provided — using random models for demo")
        return {
            'IR152':    get_model('IR_152', (img_size, img_size), emb_size).to(device),
            'IRSE50':   get_model('IRSE_50', (img_size, img_size), emb_size).to(device),
            'MobileFace': get_model('MobileFace', (img_size, img_size), emb_size).to(device),
        }

    victim_dir = Path(victim_dir)
    arch_map = {
        'IR152':     'IR_152',
        'IRSE_50':    'IRSE_50',
        'MobileFaceNet':'MobileFaceNet',
        'IR_50':      'IRSE_50',
        'IR50': 'IRSE_50',
        # 'FaceNet':   'IR_50',  # FaceNet checkpoint — using IR_50 architecture
        # 'faceNet':   'IR_50',  # Case insensitive support
    }

    models = {}
    for fname in victim_dir.glob('*.pth'):
        name = fname.stem

        try:
        # Special case: FaceNet uses Inception-ResNet-v1, not IRBackbone
            if name.lower() == 'facenet':
                m = InceptionResnetV1(pretrained=None, classify=False)
                state = torch.load(fname, map_location='cpu')
                if 'state_dict' in state:
                    state = state['state_dict']
                m.load_state_dict(state, strict=False)
                m.to(device).eval()
                models[name] = m
                print(f"  [Victim] Loaded {name} (InceptionResnetV1)")
                continue  # skip the rest of the loop for this file

            arch = arch_map.get(name, 'IR_50')
            m = get_model(arch, (img_size, img_size), emb_size)
            state = torch.load(fname, map_location='cpu')

            if 'state_dict' in state:
                state = state['state_dict']

            model_keys = set(m.state_dict().keys())
            ckpt_keys  = set(state.keys())
            print(f"  Model keys sample:      {list(model_keys)[:5]}")
            print(f"  Checkpoint keys sample: {list(ckpt_keys)[:5]}")
            print(f"  Keys in common: {len(model_keys & ckpt_keys)}")
            print(f"  Keys only in ckpt: {len(ckpt_keys - model_keys)}")

        # Strip a common "module." prefix and only keep compatible tensors.
            sanitized_state = {}

            for key, value in state.items():
                key = key[7:] if key.startswith('module.') else key

                if key in m.state_dict() and m.state_dict()[key].shape == value.shape:
                    sanitized_state[key] = value

            if not sanitized_state:
                raise ValueError("no compatible checkpoint tensors found")

            load_info = m.load_state_dict(sanitized_state, strict=False)
            m.to(device).eval()
            models[name] = m

            missing = len(load_info.missing_keys)
            unexpected = len(load_info.unexpected_keys)

            print(
                f"  [Victim] Loaded {name} ({arch}) "
                f"with {len(sanitized_state)} tensors "
                f"(missing={missing}, unexpected={unexpected})"
            )

            if missing:
                print(
                    "           Warning: checkpoint is only partially compatible;"
                    " ASR may be unreliable."
                )

        except Exception as e:
            print(f"  [Victim] Failed to load {fname}: {e}")

    if not models:
        print("[Victims] No models loaded — using random models for demo")
        models = {
            'IR152':    get_model('IR_152', (img_size, img_size), emb_size).to(device),
            'IRSE50':   get_model('IRSE_50', (img_size, img_size), emb_size).to(device),
        }
    
    print("Keys only in ckpt:")
    for k in sorted(ckpt_keys - model_keys)[:20]:
        print(f"  {k}")

    
    return models


# ──────────────────────────────────────────────────────────────────────────────
# Default thresholds (paper uses FAR@0.001 on LFW, these are typical values)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_THRESHOLDS = {
    'IRSE_50': 0.30,
    'FaceNet': 0.05,
    'IR_50':0.30,
}


# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"DPA — Diverse Parameters Augmentation Attack")
    print(f"{'='*60}")
    print(f"Device    : {device}")
    print(f"Surrogate : {args.surrogate}")
    print(f"Epochs    : {args.epochs}")
    print(f"Iterations: {args.n_iters}")
    print(f"ε         : {args.eps:.5f}")
    
    # Auto-set pretrained path if not provided
    if args.pretrained is None:
        default_path = get_default_pretrained_path(args.surrogate)
        if default_path:
            args.pretrained = default_path
            print(f"Pretrained: {args.pretrained} (auto-detected)")
        else:
            print(f"Pretrained: None (no default found for {args.surrogate})")
    else:
        print(f"Pretrained: {args.pretrained}")
    print(f"{'='*60}\n")

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_dir = os.path.join(args.output_dir, 'checkpoints')

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 1 — DPO: Build diverse surrogate set Vᶜq
    # ─────────────────────────────────────────────────────────────────────────
    if args.load_ckpts:
        print("── Step 1: Loading existing DPO checkpoints ──────────────────")
        Vq_c = load_vq_c(args.load_ckpts)
    else:
        print("── Step 1: Running DPO ───────────────────────────────────────")
        if args.train_dir is None:
            raise ValueError("--train_dir is required when not using --load_ckpts")

        train_loader, num_classes = get_train_loader(
            root        = args.train_dir,
            batch_size  = args.batch_size,
            num_workers = args.num_workers,
            img_size    = args.img_size
        )

        Vq_c = run_dpo(
            model_name      = args.surrogate,
            num_classes     = num_classes,
            dataloader      = train_loader,
            num_epochs      = args.epochs,
            lr              = args.lr,
            emb_size        = args.emb_size,
            pretrained_path = args.pretrained,
            device          = device,
            save_dir        = ckpt_dir,
            margin          = args.margin,
            scale           = args.scale,
            input_size      = (args.img_size, args.img_size),
        )

    print(f"\n  Vᶜq size: {len(Vq_c)} surrogate models\n")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 2 — HMA: Craft adversarial examples
    # ─────────────────────────────────────────────────────────────────────────
    print("── Step 2: Running HMA (crafting adversarial examples) ──────────")

    attacker = HMAAttack(
        vq_c       = Vq_c,
        model_name = args.surrogate,
        emb_size   = args.emb_size,
        n_iters    = args.n_iters,
        eps        = args.eps,
        beta       = args.beta,
        eta        = args.eta,
        layer_frac = args.layer_frac,
        device     = device,
        input_size = (args.img_size, args.img_size),
    )

    attack_loader = get_attack_loader(
        root      = args.attack_dir,
        num_pairs = args.num_pairs,
        img_size  = args.img_size
    )

    adv_examples: List[torch.Tensor] = []
    target_imgs:  List[torch.Tensor] = []

    print(f"  Attacking {args.num_pairs} source-target pairs …")
    for idx, (x_src, x_tgt) in enumerate(attack_loader):
        print(f"\n  Pair {idx+1}/{args.num_pairs}")
        x_adv = attacker.attack(x_src, x_tgt, verbose=True)
        adv_examples.append(x_adv.cpu())
        target_imgs.append(x_tgt.cpu())

    print(f"\n  Generated {len(adv_examples)} adversarial examples")

    # Save adversarial examples
    adv_save_path = os.path.join(args.output_dir, 'adv_examples.pt')
    torch.save({'adv': adv_examples, 'targets': target_imgs}, adv_save_path)
    print(f"  Saved to: {adv_save_path}")

    # ─────────────────────────────────────────────────────────────────────────
    # STEP 3 — Evaluate: Compute ASR on victim models
    # ─────────────────────────────────────────────────────────────────────────
    print("\n── Step 3: Evaluation ────────────────────────────────────────────")

    victim_models = load_victim_models(
        args.victim_dir, device, args.img_size, args.emb_size
    )

    # Use default thresholds until model-specific FAR@0.001 thresholds are available.
    thresholds = {name: DEFAULT_THRESHOLDS.get(name, 0.3)
                  for name in victim_models}

    for name, thresh in thresholds.items():
        if name in DEFAULT_THRESHOLDS:
            print(f"  [Eval] Using fallback threshold for {name}: {thresh:.4f}")
        else:
            print(f"  [Eval] Using generic fallback threshold for {name}: {thresh:.4f}")

    if args.victim_dir is None:
        print("  [Eval] Warning: victim_dir is missing, so evaluation is using demo"
              " models and the reported ASR is not meaningful.")

    evaluator = DPAEvaluator(victim_models, thresholds, device)
    results   = evaluator.evaluate(adv_examples, target_imgs)
    evaluator.print_results(results)

    # Save results
    results_path = os.path.join(args.output_dir, 'results.json')
    with open(results_path, 'w') as f:
        json.dump({k: float(v) for k, v in results.items()}, f, indent=2)
    print(f"\n  Results saved to: {results_path}")


# ──────────────────────────────────────────────────────────────────────────────
# Quick smoke test (no real data needed)
# ──────────────────────────────────────────────────────────────────────────────

def smoke_test():
    """
    Verify the full pipeline runs without errors using synthetic data.
    Does NOT produce meaningful ASR — just checks code correctness.
    """
    print("\n[Smoke Test] Running with synthetic data …")
    device = torch.device('cpu')

    # Synthetic Vᶜq (3 random surrogate state_dicts)
    Vq_c = []
    for _ in range(3):
        m = get_model('MobileFace')
        Vq_c.append(m.state_dict())

    # Build attacker
    attacker = HMAAttack(
        vq_c       = Vq_c,
        model_name = 'MobileFace',
        n_iters    = 5,      # only 5 iters for speed
        eps        = 10.0/255.0,
        beta       = 1.0/255.0,
        eta        = 1.0/255.0,
        device     = device,
    )

    # Synthetic source and target images
    x_src = torch.rand(1, 3, 112, 112)
    x_tgt = torch.rand(1, 3, 112, 112)

    x_adv = attacker.attack(x_src, x_tgt, verbose=True)

    # Verify constraint: ‖x_adv - x_src‖∞ ≤ ε
    linf = (x_adv - x_src).abs().max().item()
    print(f"\n[Smoke Test] L∞ perturbation: {linf:.5f} "
          f"(should be ≤ {10.0/255.0:.5f})")
    assert linf <= 10.0/255.0 + 1e-5, "L∞ constraint violated!"

    print("[Smoke Test] ✓ Passed — pipeline runs correctly\n")


if __name__ == '__main__':
    import sys
    if '--smoke' in sys.argv:
        smoke_test()
    else:
        main()
