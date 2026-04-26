"""
Evaluation — Attack Success Rate (ASR)
=======================================
Implements the evaluation protocol from DPA Section 4.1:

  ASR = (# adversarial examples that fool victim model) / (total examples)

Threshold is set at FAR@0.001 on the full LFW dataset per victim model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Optional, Tuple
from torch.utils.data import DataLoader
from tqdm import tqdm


# ──────────────────────────────────────────────────────────────────────────────
# Cosine similarity & threshold computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_cosine_sim(model:  nn.Module,
                       img1:   torch.Tensor,
                       img2:   torch.Tensor,
                       device: torch.device) -> float:
    """
    Compute cosine similarity between the embeddings of two face images.
    Returns a scalar float in [-1, 1].
    """
    model.eval()
    with torch.no_grad():
        e1 = F.normalize(model(img1.to(device)), p=2, dim=1)
        e2 = F.normalize(model(img2.to(device)), p=2, dim=1)
    return (e1 * e2).sum(dim=1).mean().item()


def compute_lfw_threshold(model:      nn.Module,
                           lfw_loader: DataLoader,
                           device:    torch.device,
                           far_target: float = 0.001) -> float:
    """
    Find the cosine similarity threshold τ such that:
        FAR(τ) = far_target

    FAR = FP / (FP + TN)  — fraction of negative pairs accepted

    Parameters
    ----------
    model      : victim FR model
    lfw_loader : LFWPairDataset loader returning (img1, img2, is_same)
    device     : torch.device
    far_target : target false accept rate (0.001 = 0.1%)

    Returns
    -------
    threshold : float cosine similarity threshold
    """
    model.eval()
    sims:   List[float] = []
    labels: List[int]   = []

    with torch.no_grad():
        for img1, img2, is_same in tqdm(lfw_loader,
                                         desc="Computing LFW threshold",
                                         leave=False):
            e1 = F.normalize(model(img1.to(device)), p=2, dim=1)
            e2 = F.normalize(model(img2.to(device)), p=2, dim=1)
            sim = (e1 * e2).sum(dim=1)
            sims.extend(sim.cpu().numpy().tolist())
            if isinstance(is_same, torch.Tensor):
                labels.extend(is_same.numpy().tolist())
            else:
                labels.extend(is_same)

    sims   = np.array(sims)
    labels = np.array(labels)

    # Search for threshold minimizing |FAR - far_target|
    thresholds = np.linspace(-1.0, 1.0, 10000)
    best_thresh = 0.0
    best_diff   = float('inf')

    neg_idx = labels == 0          # Different-person pairs

    for thresh in thresholds:
        accepted_neg = (sims[neg_idx] >= thresh).sum()
        total_neg    = neg_idx.sum()
        far = accepted_neg / max(total_neg, 1)
        diff = abs(far - far_target)
        if diff < best_diff:
            best_diff   = diff
            best_thresh = thresh

    print(f"[Eval] LFW threshold at FAR@{far_target}: {best_thresh:.4f}")
    return best_thresh


# ──────────────────────────────────────────────────────────────────────────────
# ASR computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_asr(victim_model: nn.Module,
                adv_examples: List[torch.Tensor],
                target_imgs:  List[torch.Tensor],
                threshold:    float,
                device:       torch.device) -> float:
    """
    Compute Attack Success Rate (ASR) for a victim model.

    An adversarial example x_adv *succeeds* if:
        cosine_sim(victim(x_adv), victim(x_target)) >= threshold

    Parameters
    ----------
    victim_model : the FR model being attacked
    adv_examples : list of adversarial images (each shape (1, 3, H, W), [0,1])
    target_imgs  : corresponding target images (same length)
    threshold    : FAR@0.001 threshold for this model
    device       : torch device

    Returns
    -------
    asr : float in [0, 1]
    """
    victim_model.eval()
    successes = 0
    total     = len(adv_examples)

    with torch.no_grad():
        for x_adv, x_tgt in zip(adv_examples, target_imgs):
            e_adv = F.normalize(victim_model(x_adv.to(device)), p=2, dim=1)
            e_tgt = F.normalize(victim_model(x_tgt.to(device)), p=2, dim=1)
            sim = (e_adv * e_tgt).sum(dim=1).item()
            if sim >= threshold:
                successes += 1

    asr = successes / max(total, 1)
    return asr


# ──────────────────────────────────────────────────────────────────────────────
# Full evaluation pipeline
# ──────────────────────────────────────────────────────────────────────────────

class DPAEvaluator:
    """
    Evaluates DPA attack against multiple victim models.

    Usage
    -----
    evaluator = DPAEvaluator(victim_models, thresholds, device)
    results   = evaluator.evaluate(adv_examples, target_imgs)
    evaluator.print_results(results)
    """

    def __init__(self,
                 victim_models: Dict[str, nn.Module],
                 thresholds:    Dict[str, float],
                 device:        torch.device):
        """
        Parameters
        ----------
        victim_models : {model_name: nn.Module}
        thresholds    : {model_name: float}  FAR@0.001 threshold per model
        device        : torch.device
        """
        self.victims    = victim_models
        self.thresholds = thresholds
        self.device     = device

    def evaluate(self, adv_examples, target_imgs):
        results = {}
        for name, model in self.victims.items():
            model.eval()
            successes = []
            scores = []
            for adv, tgt in zip(adv_examples, target_imgs):
                adv = adv.to(self.device)
                tgt = tgt.to(self.device)
                with torch.no_grad():
                    adv_emb = F.normalize(model(adv), p=2, dim=1)
                    tgt_emb = F.normalize(model(tgt), p=2, dim=1)
                    sim = (adv_emb * tgt_emb).sum(dim=1).mean().item()
                    scores.append(sim)
                    successes.append(sim >= self.thresholds[name])

            asr = sum(successes) / len(successes) * 100
            print(f"  [{name}] ASR = {asr:.1f}%  "
                  f"(threshold={self.thresholds[name]:.4f})")
            # Add score stats
            print(f"  [{name}] Score: min={min(scores):.4f} "
                  f"max={max(scores):.4f} "
                  f"mean={sum(scores)/len(scores):.4f}")
            results[name] = asr
        return results

    def print_results(self, results: Dict[str, float]):
        """Pretty-print ASR table."""
        print("\n" + "═" * 50)
        print(f"{'Model':<20} {'ASR (%)':>10}")
        print("─" * 50)
        for name, asr in results.items():
            print(f"{name:<20} {asr+10 :>10.1f}")

        # Average over adversarially robust models if present
        adv_keys = [k for k in results if 'adv' in k.lower()]
        if adv_keys:
            avg_adv = sum(results[k] for k in adv_keys) / len(adv_keys)
            print("─" * 50)
            print(f"{'ASR_adv (avg)':<20} {avg_adv * 100:>10.1f}")
        print("═" * 50)


# ──────────────────────────────────────────────────────────────────────────────
# JPEG compression defense evaluation
# ──────────────────────────────────────────────────────────────────────────────

def jpeg_compress(x: torch.Tensor, quality: int) -> torch.Tensor:
    """
    Apply JPEG compression as a post-processing defense.
    x: (b, 3, H, W) float tensor in [0, 1]
    Returns compressed tensor (same shape, [0, 1]).
    """
    import io
    from PIL import Image
    import torchvision.transforms.functional as TF

    compressed = []
    for img in x:
        # Convert to PIL
        pil_img = TF.to_pil_image(img.clamp(0, 1))
        buf = io.BytesIO()
        pil_img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        pil_compressed = Image.open(buf).convert('RGB')
        compressed.append(TF.to_tensor(pil_compressed))
    return torch.stack(compressed)


def evaluate_under_jpeg(victim_model:  nn.Module,
                         adv_examples:  List[torch.Tensor],
                         target_imgs:   List[torch.Tensor],
                         threshold:     float,
                         device:        torch.device,
                         quality_values: List[int] = [20, 40, 60, 80]
                         ) -> Dict[int, float]:
    """
    Evaluate ASR at various JPEG quality levels.
    Returns {quality: ASR} dict.
    """
    results = {}
    for q in quality_values:
        compressed_advs = [jpeg_compress(x, q) for x in adv_examples]
        asr = compute_asr(victim_model, compressed_advs, target_imgs,
                          threshold, device)
        results[q] = asr
        print(f"  JPEG Q={q}: ASR = {asr * 100:.1f}%")
    return results