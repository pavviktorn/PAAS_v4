#!/usr/bin/env python3.12
"""Self-test of the GSD implementation (no training data needed). Checks the math the paper defines:

  * U is orthonormal:           U^T U == I_K
  * de-semanticized features are orthogonal to the basis:  F'(I-UU^T) @ U == 0
  * frozen stream carries no gradient; trainable forward produces (B,) logits
  * GSD actually changed the features (projection is non-trivial)
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsd.config import GSDConfig
from gsd.model import GSDDetector
from gsd.householder import semantic_basis
from gsd.projection import desemanticize


def main():
    torch.manual_seed(0)
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"

    # 1) pure-math checks on the basis + projection (fast, no model)
    D, B, K = 1024, 32, 16
    guide = torch.randn(B, D)
    U = semantic_basis(guide, K, "householder")
    assert U.shape == (D, K), U.shape
    ortho = (U.t() @ U - torch.eye(K)).abs().max().item()
    F = torch.randn(B, 50, D)
    Fp = desemanticize(F, U)
    resid = (Fp @ U).abs().max().item()                 # must be ~0 (orthogonal complement)
    changed = (Fp - F).abs().max().item()
    print(f"[math] U^T U == I  max|err|={ortho:.2e}  | F'(.)@U max|.|={resid:.2e}  | changed={changed:.3f}")
    assert ortho < 1e-4 and resid < 1e-3 and changed > 1e-3

    # SVD variant should also satisfy orthogonality
    Us = semantic_basis(guide, K, "svd")
    assert (Us.t() @ Us - torch.eye(K)).abs().max().item() < 1e-4

    # 2) full model forward (loads the real CLIP) + gradient sanity
    cfg = GSDConfig(k=K, n_gsd_layers=4, trainable="head")   # head-only keeps it light for the test
    print(f"[model] building dual-stream CLIP from {cfg.clip_path} ...", flush=True)
    model = GSDDetector(cfg).to(dev)
    x = torch.randn(8, 3, cfg.image_size, cfg.image_size, device=dev)
    logits = model(x)
    assert logits.shape == (8, cfg.num_classes), logits.shape
    assert all(not p.requires_grad for p in model.frozen.parameters())
    layers = model.trainable.vision_model.encoder.layers
    used = [i for i in model.gsd_layer_ids if getattr(layers[i], "_gsd_U", None) is not None]
    print(f"[model] forward OK: logits {tuple(logits.shape)} ({cfg.num_classes}-class) | "
          f"GSD-projected layers: {used}")

    # 3) backward reaches the head (trainable path differentiable through the projection)
    y = torch.randint(0, cfg.num_classes, (8,), device=dev)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    g = model.head[-1].weight.grad
    assert g is not None and torch.isfinite(g).all()
    print(f"[grad] head grad finite, norm={g.norm().item():.4f}")
    print("\nALL GSD CHECKS PASSED ✅")


if __name__ == "__main__":
    main()
