"""Geometry helpers for ADRiver-4D (kNN on point sets, local statistics)."""

from __future__ import annotations

from typing import Tuple

import torch


def knn_indices_self(xyz: torch.Tensor, k: int) -> torch.Tensor:
    """
    xyz: [BL, N, 3] -> idx: [BL, N, k], exclude self-neighbor.
    Pure PyTorch (cdist); O(N^2) memory — use at moderate N (encoder bottlenecks).
    """
    bl, n, _ = xyz.shape
    if n <= 1:
        return torch.zeros(bl, n, k, device=xyz.device, dtype=torch.long)

    dist = torch.cdist(xyz, xyz, p=2)
    diag = torch.eye(n, device=xyz.device, dtype=xyz.dtype).unsqueeze(0).expand(bl, -1, -1)
    dist = dist + diag * 1e4
    kn = min(k, max(n - 1, 1))
    _, idx = dist.topk(kn, largest=False, dim=-1)
    if kn < k:
        pad = idx[..., -1:].expand(-1, -1, k - kn)
        idx = torch.cat([idx, pad], dim=-1)
    return idx[..., :k]


def gather_neighbors(
    flat_feat: torch.Tensor,
    flat_xyz: torch.Tensor,
    idx: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    flat_feat: [BL, N, C], flat_xyz: [BL, N, 3], idx: [BL, N, K]
    -> feat_nei: [BL, N, K, C], xyz_nei: [BL, N, K, 3]
    """
    bl, n, k = idx.shape
    c = flat_feat.shape[-1]
    bl_ar = torch.arange(bl, device=flat_feat.device, dtype=torch.long).view(bl, 1, 1).expand(-1, n, k)
    feat_nei = flat_feat[bl_ar, idx, :]
    xyz_nei = flat_xyz[bl_ar, idx, :]
    return feat_nei, xyz_nei


def soft_knn_interpolate_vectors(
    xyz_source: torch.Tensor,
    vec_source: torch.Tensor,
    xyz_target: torch.Tensor,
    k: int = 3,
    sigma: float = 0.1,
) -> torch.Tensor:
    """
    For each target point, interpolate D-dim vectors from source using softmax(-dist/sigma).

    xyz_source: [B, L, Ns, 3]
    vec_source: [B, L, Ns, D]
    xyz_target: [B, L, Nt, 3]
    -> [B, L, Nt, D]
    """
    bsz, l, ns, _ = xyz_source.shape
    nt = xyz_target.shape[2]
    d = vec_source.shape[-1]
    bl = bsz * l
    xs = xyz_source.reshape(bl, ns, 3)
    vs = vec_source.reshape(bl, ns, d)
    xt = xyz_target.reshape(bl, nt, 3)
    dist = torch.cdist(xt, xs, p=2)
    kk = min(k, ns)
    val, idx = dist.topk(kk, largest=False, dim=-1)
    w = torch.softmax(-val / max(float(sigma), 1e-8), dim=-1)
    bl_range = torch.arange(bl, device=vec_source.device, dtype=torch.long).view(bl, 1, 1).expand(bl, nt, kk)
    gathered = vs[bl_range, idx, :]
    out = (w.unsqueeze(-1) * gathered).sum(dim=2)
    return out.reshape(bsz, l, nt, d)
