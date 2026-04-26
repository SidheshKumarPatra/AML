"""
Face Recognition Head — ArcFace (Additive Angular Margin)
Implements exactly the loss described in Section 3.2 of DPA:
  cos(a + m) = cos(a)cos(m) - sin(a)sin(m)
  q = d * (h_hat ⊙ p + (1 - h_hat) ⊙ r)
  L = CrossEntropy(q, y)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ArcFaceHead(nn.Module):
    """
    ArcFace classification head.

    Parameters
    ----------
    emb_size  : embedding dimension (r in the paper), e.g. 512
    num_classes : number of identities in training set (s in the paper)
    margin    : angular margin m, default 0.5 rad ≈ 28.6°
    scale     : scale factor d, default 64
    """

    def __init__(self, emb_size: int, num_classes: int,
                 margin: float = 0.5, scale: float = 64.0):
        super().__init__()
        self.emb_size    = emb_size
        self.num_classes = num_classes
        self.margin      = margin
        self.scale       = scale

        # Head weight matrix W ∈ R^{s × r}
        # Each row is the class prototype (normalized during forward)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        # Precompute margin constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        # Safe lower bound: cos(π - m)
        self.threshold = math.cos(math.pi - margin)
        # Fallback for angle ≥ π - m: cos(a) - m·sin(m)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        embeddings : (b, r)  — L2-normalised face embeddings
        labels     : (b,)    — ground-truth identity indices

        Returns
        -------
        loss : scalar cross-entropy loss
        """
        # ── Step 1: normalize embeddings and weights ──────────────────
        e_norm = F.normalize(embeddings, p=2, dim=1)          # ē  (b, r)
        w_norm = F.normalize(self.weight, p=2, dim=1)         # w̄  (s, r)

        # ── Step 2: cosine similarity matrix r = ē·w̄ᵀ ───────────────
        cos_a = e_norm @ w_norm.t()                            # (b, s)

        # ── Step 3: sine similarity  sin(a) = √(1 - cos²(a)) ─────────
        # Clamp cos_a to [-1, 1] first to avoid numerical issues
        cos_a_clamped = torch.clamp(cos_a, -1.0, 1.0)
        sin_a = torch.sqrt(
            (1.0 - cos_a_clamped.pow(2)).clamp(min=1e-9)
        )                                                      # (b, s)

        # ── Step 4: additive angular margin  cos(a + m) ───────────────
        # cos(a+m) = cos(a)·cos(m) - sin(a)·sin(m)
        # Use clamped cos_a for stability
        cos_a_m = cos_a_clamped * self.cos_m - sin_a * self.sin_m     # (b, s)

        # Fallback for numerical stability when a ≥ π - m
        # p = cos(a+m)           if a < π - m
        # p = cos(a) - m·sin(m)  if a ≥ π - m
        p = torch.where(
            cos_a_clamped > self.threshold,
            cos_a_m,
            cos_a_clamped - self.mm
        )                                                      # (b, s)

        # ── Step 5: one-hot encode labels ĥ ──────────────────────────
        h_hat = torch.zeros_like(cos_a)                       # (b, s)
        h_hat.scatter_(1, labels.view(-1, 1), 1.0)

        # ── Step 6: head output q = d·(ĥ⊙p + (1-ĥ)⊙r) ───────────────
        q = self.scale * (h_hat * p + (1.0 - h_hat) * cos_a)  # (b, s)

        # ── Step 7: cross-entropy loss ────────────────────────────────
        loss = F.cross_entropy(q, labels)
        return loss

    def get_cosine_similarity(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Returns raw cosine similarity matrix (no margin) for inference.
        Used during adversarial attack to measure embedding distance.
        """
        e_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        return e_norm @ w_norm.t()


class CosineLoss(nn.Module):
    """
    Cosine softmax head (no angular margin).
    Used as a simpler baseline / alternative head.
    """
    def __init__(self, emb_size: int, num_classes: int, scale: float = 64.0):
        super().__init__()
        self.scale  = scale
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, emb_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, embeddings: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        e_norm = F.normalize(embeddings, p=2, dim=1)
        w_norm = F.normalize(self.weight, p=2, dim=1)
        logits = self.scale * (e_norm @ w_norm.t())
        return F.cross_entropy(logits, labels)