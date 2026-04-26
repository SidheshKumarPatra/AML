"""
Stage 2 — Hard Model Aggregation (HMA) + Adversarial Attack
============================================================
Implements Algorithm 2 from the DPA paper exactly.

Key ideas:
  1. Map Vᶜq → F = {F₁, ..., Fg}  (g surrogate models)
  2. For each iteration t:
       a. Apply input transform T to x_adv_{t-1}
       b. Forward through each Fᵢ layer-by-layer
       c. At each layer in Φ (t > 1): add beneficial perturbation
              ω_{t,j} += η·sign(∇_{ω_{t-1,j}} L̃_{t-1})
       d. Compute loss: L̃ᵢ = ‖ϕ(Fᵢ^{Φu,z}(ω_{t,u})) - ϕ(Fᵢ(xᵗ))‖²
       e. Average: L̃_t = (1/g)·ΣL̃ᵢ
       f. Update: x_adv_t = Π_{xˢ,ε}(x_adv_{t-1} - β·sign(∇_{x_adv} L̃_t))
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import random
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from models.backbones import get_model


# ──────────────────────────────────────────────────────────────────────────────
# Input transformation T  (random resizing + padding, from DI paper)
# ──────────────────────────────────────────────────────────────────────────────

def input_transform(x: torch.Tensor,
                    resize_range: Tuple[int, int] = (90, 112),
                    output_size: int = 112,
                    prob: float = 0.7) -> torch.Tensor:
    if random.random() > prob:
        return x

    b, c, h, w = x.shape
    rnd = random.randint(resize_range[0], resize_range[1])

    # Keep gradient graph intact
    x_resized = F.interpolate(x, size=(rnd, rnd),
                               mode='bilinear', align_corners=False,
                               recompute_scale_factor=False)  # ← add this

    pad_h = output_size - rnd
    pad_w = output_size - rnd
    pad_top    = random.randint(0, pad_h)
    pad_bottom = pad_h - pad_top
    pad_left   = random.randint(0, pad_w)
    pad_right  = pad_w - pad_left

    x_padded = F.pad(x_resized,
                     (pad_left, pad_right, pad_top, pad_bottom),
                     mode='constant', value=0)
    return x_padded

# ──────────────────────────────────────────────────────────────────────────────
# Layer-wise forward pass (needed for feature-map perturbation)
# ──────────────────────────────────────────────────────────────────────────────

def get_named_layers(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """
    Returns a flat list of (name, layer) for all leaf modules in model.
    Defines the layer index set used for beneficial perturbation.
    """
    layers = []
    for name, module in model.named_modules():
        # Keep only leaf modules (actual computation nodes)
        if len(list(module.children())) == 0:
            layers.append((name, module))
    return layers


def select_perturbation_layers(model: nn.Module,
                                layer_frac: float = 0.3) -> List[int]:
    """
    Select which layer indices to add beneficial perturbations to.
    DPA uses the intermediate conv/BN layers (Ω).
    We select the top `layer_frac` fraction of layers by depth.

    Returns list of integer layer indices into the flat layer list.
    """
    layers = get_named_layers(model)
    total  = len(layers)
    # Select layers in the middle portion of the network
    start  = int(total * 0.3)
    end    = int(total * 0.3 + total * layer_frac)
    return list(range(start, min(end, total)))


# ──────────────────────────────────────────────────────────────────────────────
# Hard model forward pass (Algorithm 2, inner loop)
# ──────────────────────────────────────────────────────────────────────────────

class HardModelForward:
    """
    Stateful object that wraps a surrogate model and maintains the
    beneficial perturbation buffers ω for each perturbed layer.

    On the first iteration (t=1): plain forward pass, no perturbation.
    On subsequent iterations: adds η·sign(∇_ω L̃_{t-1}) to feature maps
    at the selected layers.
    """

    def __init__(self,
                 model: nn.Module,
                 perturb_layer_indices: List[int],
                 eta: float,
                 device: torch.device):
        self.model   = model.to(device).eval()
        self.phi_idx = perturb_layer_indices   # Φ — selected layer indices
        self.eta     = eta
        self.device  = device

        # Flat list of all leaf layers
        self._layers = get_named_layers(model)

        # ω buffers — one per perturbed layer, initialized to None
        # Shape will be set on first forward pass
        self.omega: List[Optional[torch.Tensor]] = [None] * len(self.phi_idx)

        # Store feature-map values from last forward (needed for gradient)
        self._last_features: List[Optional[torch.Tensor]] = \
            [None] * len(self.phi_idx)

    def reset(self):
        """Reset perturbation buffers (call between different adversarial images)."""
        self.omega = [None] * len(self.phi_idx)
        self._last_features = [None] * len(self.phi_idx)

    def forward(self,
                x: torch.Tensor,
                prev_loss: Optional[torch.Tensor],
                is_first_iter: bool) -> torch.Tensor:
        """
        Hard model forward pass — χ(F(x_adv_t), Ω).

        Parameters
        ----------
        x              : (b, 3, H, W) input (already transformed by T)
        prev_loss      : L̃_{t-1} scalar loss (used to compute ∇_ω)
        is_first_iter  : True if t == 1 (no perturbation applied)

        Returns
        -------
        embedding : (b, emb_size) output from final layer
        """
        # Update ω buffers using gradient of previous loss
        # if not is_first_iter and prev_loss is not None:
        #     self._update_omega(prev_loss)

        # Layer-wise forward with perturbation injection
        embedding = self.model(x)
        return embedding

    def _update_omega(self, prev_loss: torch.Tensor):
        """
        ω_{t,j} = ω_{t-1,j} + η·sign(∇_{ω_{t-1,j}} L̃_{t-1})
        Equation (15) in DPA.
        """
        for j, feat in enumerate(self._last_features):
            if feat is None or not feat.requires_grad:
                continue
            if feat.grad is None:
                continue
            grad_sign = feat.grad.data.sign()
            if self.omega[j] is None:
                self.omega[j] = (self.eta * grad_sign).detach()
            else:
                self.omega[j] = (self.omega[j] + self.eta * grad_sign).detach()

    def _update_omega_from_store(self):
        """
        Called immediately after backward() while grads are still alive.
        Updates omega buffers using retained gradients on stored features.
        """
        for j, feat in enumerate(self._last_features):
            if feat is None:
                continue
            if feat.grad is None:
                continue
            grad_sign = feat.grad.data.sign()
            if self.omega[j] is None:
                self.omega[j] = (self.eta * grad_sign).detach()
            else:
                self.omega[j] = (self.omega[j] + self.eta * grad_sign).detach()

    def _layerwise_forward(self, x: torch.Tensor,
                            apply_perturbation: bool) -> torch.Tensor:
        """
        Passes x through the model layer by layer.
        At layers in Φ, injects the stored ω perturbation.

        The feature map at each perturbed layer is saved with
        requires_grad=True so we can compute ∇_ω L̃_{t}.
        """
        # Reset gradient tracking on feature store
        self._last_features = [None] * len(self.phi_idx)

        # Build a set of target layer indices for O(1) lookup
        phi_set = {self.phi_idx[j]: j for j in range(len(self.phi_idx))}

        h = x
        layer_counter = 0

        # We need to run through the model's modules in order.
        # We reconstruct the forward pass by calling the model's
        # sequential sub-modules manually.
        h = self._forward_backbone(h, phi_set, apply_perturbation)
        return h

    def _forward_backbone(self,
                           x: torch.Tensor,
                           phi_set: Dict[int, int],
                           apply_perturbation: bool) -> torch.Tensor:
        """
        Walk through the model, injecting perturbations at phi_set layers.
        This works for any nn.Module by hooking into the forward hooks.
        """
        feat_store = {}
        hooks = []

        def make_pre_hook(layer_idx):
            def hook(module, inputs):
                inp = inputs[0]
                j_idx = phi_set.get(layer_idx, None)
                if j_idx is not None and apply_perturbation:
                    if self.omega[j_idx] is not None:
                        # Add beneficial perturbation to input feature map
                        omega = self.omega[j_idx]
                        if omega.shape == inp.shape:
                            inp = inp + omega.to(inp.device)
                    # Track for gradient computation
                    # inp = inp.detach().requires_grad_(True)
                    # ── FIX: retain grad so _update_omega can read it ──
                    inp = inp.detach().requires_grad_(True)
                    inp.retain_grad()
                    feat_store[layer_idx] = inp
                    return (inp,) + inputs[1:]
            return hook 

        # Register pre-forward hooks on all leaf layers
        layers = get_named_layers(self.model)
        for idx, (name, module) in enumerate(layers):
            if idx in phi_set:
                h = module.register_forward_pre_hook(make_pre_hook(idx))
                hooks.append(h)

        # Run full forward pass — grad must be enabled for x_adv gradient
        output = self.model(x)

        # Remove hooks
        for h in hooks:
            h.remove()

        # Store tracked features (for gradient computation next iteration)
        for layer_idx, feat in feat_store.items():
            j_idx = phi_set[layer_idx]
            self._last_features[j_idx] = feat

        return output


# ──────────────────────────────────────────────────────────────────────────────
# HMA Attack — main class (Algorithm 2)
# ──────────────────────────────────────────────────────────────────────────────

class HMAAttack:
    """
    DPA Attack — combines DPO surrogate set with HMA adversarial optimization.

    Parameters
    ----------
    vq_c         : List of backbone state_dicts (output of DPO)
    model_name   : backbone architecture string
    emb_size     : embedding dimension
    n_iters      : maximum attack iterations (200 in paper)
    eps          : L∞ perturbation budget (10/255 or 10 raw pixels)
    beta         : adversarial step size
    eta          : beneficial perturbation step size
    layer_frac   : fraction of layers used for beneficial perturbation
    device       : torch.device
    input_size   : (H, W) of face crops
    """

    def __init__(self,
                 vq_c:        List[Dict],
                 model_name:  str,
                 emb_size:    int          = 512,
                 n_iters:     int          = 200,
                 eps:         float        = 10.0 / 255.0,
                 beta:        float        = 1.0 / 255.0,
                 eta:         float        = 1.0 / 255.0,
                 layer_frac:  float        = 0.3,
                 device:      Optional[torch.device] = None,
                 input_size:  Tuple[int, int] = (112, 112)):

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.n_iters    = n_iters
        self.eps        = eps
        self.beta       = beta
        self.eta        = eta
        self.device     = device
        self.input_size = input_size

        print(f"[HMA] Loading {len(vq_c)} surrogate models …")
        self.models: List[nn.Module] = []
        for idx, state_dict in enumerate(vq_c):
            m = get_model(model_name, input_size=input_size, emb_size=emb_size)
            m.load_state_dict(state_dict)
            m.to(device).eval()
            self.models.append(m)
            
        # Pre-select perturbation layer indices (same set for all models)
        ref_model = self.models[0]
        self.phi = select_perturbation_layers(ref_model, layer_frac)
        print(f"[HMA] g={len(self.models)} surrogates, "
              f"|Φ|={len(self.phi)} perturbed layers")

        
        # # In HMAAttack.__init__, after loading models
        # print("[HMA] Filtering surrogate quality...")
        # good_models = []
        # test_img1 = torch.randn(2, 3, 112, 112).to(device)
        # test_img2 = torch.randn(2, 3, 112, 112).to(device)

        # for i, m in enumerate(self.models):
        #     with torch.no_grad():
        #         e1 = F.normalize(m(test_img1), p=2, dim=1)
        #         e2 = F.normalize(m(test_img2), p=2, dim=1)
        #         sim = (e1 * e2).sum().item()
        #     if sim < 0.95:  # good surrogate can distinguish random faces
        #         good_models.append(m)
        #         print(f"  Surrogate {i}: sim={sim:.4f} ✓ kept")
        #     else:
        #         print(f"  Surrogate {i}: sim={sim:.4f} ✗ filtered (too similar)")

        # self.models = good_models if good_models else self.models
        # print(f"[HMA] Using {len(self.models)} good surrogates")
        
        # Build HardModelForward wrappers
        self.hard_models: List[HardModelForward] = [
            HardModelForward(m, self.phi, eta, device)
            for m in self.models
        ]

    def _normalize_emb(self, emb: torch.Tensor) -> torch.Tensor:
        """ϕ(·) — L2 normalize embedding."""
        return F.normalize(emb, p=2, dim=1)

    def _compute_target_embeddings(self,
                                    x_t: torch.Tensor) -> List[torch.Tensor]:
        """
        Pre-compute ϕ(Fᵢ(xᵗ)) for all g models — constant across iterations.
        """
        target_embs = []
        for model in self.models:
            with torch.no_grad():
                emb = self._normalize_emb(model(x_t))
                target_embs.append(emb.detach())
        return target_embs

    def attack(self,
               x_source: torch.Tensor,
               x_target: torch.Tensor,
               verbose:  bool = True) -> torch.Tensor:
        """
        Craft adversarial example x_adv starting from x_source,
        targeting x_target's identity.

        Parameters
        ----------
        x_source : (1 or b, 3, H, W)  source face, values in [-1, 1]
        x_target : (1 or b, 3, H, W)  target face, values in [-1, 1]
        verbose  : print per-iteration loss

        Returns
        -------
        x_adv : (b, 3, H, W) adversarial face, values in [-1, 1]
        """
        x_source = x_source.to(self.device)
        x_target = x_target.to(self.device)

        # Pre-compute target embeddings (Equation 13, constant term)
        target_embs = self._compute_target_embeddings(x_target)

        # Initialise x_adv = x_source
        x_adv = x_source.clone().detach().requires_grad_(True)

        # Reset hard model buffers
        for hm in self.hard_models:
            hm.reset()

        prev_loss: Optional[torch.Tensor] = None

        pbar = tqdm(range(1, self.n_iters + 1),
                    desc="[HMA] Attack", disable=not verbose)

        for t in pbar:
            # Ensure x_adv is always a proper leaf tensor
            assert x_adv.requires_grad, f"t={t}: x_adv lost requires_grad!"
            assert x_adv.is_leaf, f"t={t}: x_adv is not a leaf tensor!"

            # ── Apply input transformation T ──────────────────────────
            x_t = input_transform(x_adv,
                                   output_size=self.input_size[0])

            # Ensure gradient flows back to x_adv
            if not x_t.requires_grad:
                x_t = x_t.requires_grad_(True)

            # ── Compute aggregated loss L̃_t ───────────────────────────
            total_loss = None
            g = len(self.hard_models)

            for i, hm in enumerate(self.hard_models):
                # Hard model forward: Hᵢ(T(x_adv_t))
                adv_emb = hm.forward(x_t, prev_loss, is_first_iter=(t == 1))
                adv_emb_norm = self._normalize_emb(adv_emb)

                # L̃ᵢ = ‖ϕ(Hᵢ(T(x_adv))) - ϕ(Fᵢ(xᵗ))‖²
                loss_i = torch.norm(adv_emb_norm - target_embs[i], p=2) ** 2
                if total_loss is None:
                    total_loss = loss_i / g
                else:
                    total_loss = total_loss + loss_i / g

            # ── Backward to get ∇_{x_adv} L̃_t ───────────────────────
            total_loss.sum().backward()
            # ── Update omega RIGHT NOW while grads still alive ────────
            if t > 1:
                for hm in self.hard_models:
                    hm._update_omega_from_store()

            with torch.no_grad():
                # ── Guard against None grad ───────────────────────────
                if x_adv.grad is None:
                    print(f"  [Warning] t={t}: x_adv.grad is None, skipping update")
                    x_adv = x_adv.detach().requires_grad_(True)
                    prev_loss = total_loss.detach()
                    continue

                # ── Sign gradient update ──────────────────────────────
                # x_adv_{t+1} = Π_{xˢ,ε}(x_adv_t - β·sign(∇_{x_adv} L̃_t))
                grad_sign = x_adv.grad.data.sign()
                x_adv_new = x_adv.detach() - self.beta * grad_sign

                # ── Project back to L∞ ε-ball around x_source ────────
                delta = torch.clamp(x_adv_new - x_source,
                                    min=-self.eps, max=self.eps)
                # Inputs arrive already normalized for the FR models.
                x_adv_new = torch.clamp(x_source + delta, min=-1.0, max=1.0)
            
            # Detach and create new leaf tensor for next iteration
            x_adv = x_adv_new.detach().requires_grad_(True)

            prev_loss = total_loss.detach()

            if verbose and (t % 5 == 0 or t == 1):
                pbar.set_postfix(loss=f"{total_loss.item():.4f}")
# Debug: check actual similarity scores
        with torch.no_grad():
            print(f"\n  [Debug] Perturbation L∞: {(x_adv - x_source).abs().max().item():.4f}")
            for i, model in enumerate(self.models):
                adv_emb = F.normalize(model(x_adv), p=2, dim=1)
                tgt_emb = F.normalize(model(x_target), p=2, dim=1)
                src_emb = F.normalize(model(x_source), p=2, dim=1)
                sim_adv_tgt = (adv_emb * tgt_emb).sum(dim=1).item()
                sim_src_tgt = (src_emb * tgt_emb).sum(dim=1).item()
                print(f"  [Debug] Surrogate {i}: sim(adv,tgt)={sim_adv_tgt:.4f} sim(src,tgt)={sim_src_tgt:.4f}")
        return x_adv.detach()


# ──────────────────────────────────────────────────────────────────────────────
# Cosine similarity utility (used in evaluation)
# ──────────────────────────────────────────────────────────────────────────────

def cosine_similarity(emb1: torch.Tensor, emb2: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two (b, d) embedding tensors."""
    e1 = F.normalize(emb1, p=2, dim=1)
    e2 = F.normalize(emb2, p=2, dim=1)
    return (e1 * e2).sum(dim=1)
