"""
ADRiver-4D dynamics: one explicit ADR step in feature space.

A: motion-aligned soft advection (scene-flow-conditioned neighbor mixing)
D: graph diffusion (Laplacian) with learnable magnitude D(x)
R: reaction MLP on [h, flow, local geometry statistics]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .geometry import gather_neighbors, knn_indices_self


class ADRiverDynamics(nn.Module):
    """
    Single Euler update: h <- h + dt * (adv + D*Lap h + R(h,...))

    f_seq: [B, L, N, C]
    xyz:   [B, L, N, 3]  (aligned with f_seq)
    """

    def __init__(
        self,
        c_dim: int,
        k_neighbors: int = 16,
        advection_tau: float = 0.15,
        learnable_dt: bool = True,
        dt_init: float = 1.0,
    ):
        super().__init__()
        self.c_dim = c_dim
        self.k_neighbors = k_neighbors
        self.advection_tau = advection_tau

        self.flow_head = nn.Conv1d(c_dim, 3, kernel_size=1)
        self.diffusion_head = nn.Conv1d(c_dim, 1, kernel_size=1)
        self.uncertainty_head = nn.Conv1d(c_dim, 1, kernel_size=1)
        self.global_flow_proj = nn.Sequential(
            nn.Linear(3, c_dim),
            nn.ReLU(inplace=True),
            nn.Linear(c_dim, c_dim),
        )
        self.global_adv_gate = nn.Sequential(
            nn.Conv1d(c_dim, c_dim, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        # R: h, flow(3), mean_dist(1), var_dist(1) -> c_dim
        r_in = c_dim + 3 + 1 + 1
        self.reaction_mlp = nn.Sequential(
            nn.Conv1d(r_in, c_dim, kernel_size=1, bias=True),
            nn.BatchNorm1d(c_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(c_dim, c_dim, kernel_size=1, bias=True),
        )

        if learnable_dt:
            self.log_dt = nn.Parameter(torch.tensor(dt_init).log())
        else:
            self.register_buffer("log_dt", torch.tensor(dt_init).log(), persistent=False)

    def dt(self) -> torch.Tensor:
        return self.log_dt.exp().clamp(1e-4, 10.0)

    def forward(
        self,
        f_seq: torch.Tensor,
        xyz: torch.Tensor,
        uncertainty_prior: torch.Tensor | None = None,
        return_components: bool = False,
    ):
        B, L, N, C = f_seq.shape
        bl = B * L
        f_flat = f_seq.permute(0, 1, 3, 2).reshape(bl, C, N)
        flow_pred = (
            self.flow_head(f_flat).reshape(B, L, 3, N).permute(0, 1, 3, 2).contiguous()
        )
        diffusion = F.softplus(self.diffusion_head(f_flat)).reshape(B, L, 1, N).permute(0, 1, 3, 2)
        uncertainty = torch.sigmoid(
            self.uncertainty_head(f_flat).reshape(B, L, 1, N).permute(0, 1, 3, 2)
        )
        if uncertainty_prior is not None:
            uncertainty_prior = uncertainty_prior.clamp(0.0, 1.0)
            uncertainty = 0.5 * (uncertainty + uncertainty_prior)

        xyz_bl = xyz.reshape(bl, N, 3)
        f_bl = f_seq.reshape(bl, N, C)
        idx = knn_indices_self(xyz_bl, self.k_neighbors)
        feat_nei, xyz_nei = gather_neighbors(f_bl, xyz_bl, idx)

        f_nei_b = feat_nei.reshape(B, L, N, self.k_neighbors, C)
        xyz_nei_b = xyz_nei.reshape(B, L, N, self.k_neighbors, 3)

        neigh_mean = f_nei_b.mean(dim=3)
        lap_spatial = neigh_mean - f_seq
        diffusion_eff = diffusion * (1.0 + uncertainty)
        diff_term = diffusion_eff * lap_spatial

        delta = xyz_nei_b - xyz.unsqueeze(3)
        v_dir = F.normalize(flow_pred.unsqueeze(3), dim=-1, eps=1e-6)
        d_norm = F.normalize(delta, dim=-1, eps=1e-6)
        cos_align = (d_norm * v_dir).sum(dim=-1)
        w = F.softmax(cos_align / self.advection_tau, dim=-1)
        h_flow = (w.unsqueeze(-1) * f_nei_b).sum(dim=3)
        adv_feat = h_flow - f_seq
        flow_global = flow_pred.mean(dim=2)  # [B, L, 3]
        flow_global_feat = self.global_flow_proj(flow_global).unsqueeze(2)  # [B, L, 1, C]
        adv_gate = self.global_adv_gate(
            f_seq.permute(0, 1, 3, 2).reshape(bl, C, N)
        ).reshape(B, L, C, N).permute(0, 1, 3, 2)
        adv_global = adv_gate * flow_global_feat
        adv_feat = adv_feat + adv_global

        dists = delta.norm(dim=-1)
        neigh_dist = dists.mean(dim=-1, keepdim=True)
        neigh_var = dists.var(dim=-1, keepdim=True, unbiased=False)

        r_cat = torch.cat([f_seq, flow_pred, neigh_dist, neigh_var], dim=-1)
        r_in = r_cat.permute(0, 1, 3, 2).reshape(bl, r_cat.shape[-1], N)
        reac_term = (
            self.reaction_mlp(r_in).reshape(B, L, C, N).permute(0, 1, 3, 2).contiguous()
        )

        dt = self.dt()
        f_next = f_seq + dt * (adv_feat + diff_term + reac_term)

        out_components = {
            "flow_pred": flow_pred,
            "adv_feat": adv_feat,
            "lap_spatial": lap_spatial,
            "diffusion": diffusion,
            "diffusion_eff": diffusion_eff,
            "uncertainty": uncertainty,
            "diff_term": diff_term,
            "reaction": reac_term,
            "advection_weights": w,
            "adv_global": adv_global,
            "dt": dt,
        }
        if return_components:
            return f_next, out_components
        return f_next


class ADRiverRefinement(nn.Module):
    """K repeated ADR dynamics with shared weights (unrolled iterative refinement)."""

    def __init__(self, dynamics: ADRiverDynamics, num_iters: int = 3):
        super().__init__()
        self.dynamics = dynamics
        self.num_iters = num_iters

    def forward(
        self,
        f_seq: torch.Tensor,
        xyz: torch.Tensor,
        uncertainty_prior: torch.Tensor | None = None,
    ):
        aux_all = []
        h = f_seq
        for _ in range(self.num_iters):
            h, comp = self.dynamics(
                h,
                xyz,
                uncertainty_prior=uncertainty_prior,
                return_components=True,
            )
            aux_all.append(comp)
        return h, aux_all
