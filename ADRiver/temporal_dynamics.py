"""
Temporal ADR along the frame axis (same slot index n at bottleneck).
Shares the *semantic* structure of spatial ADRiverDynamics (A + D + R) with shared tau / dt style.

Operates on f: [B, L, N, C], xyz: [B, L, N, 3] *before* or *after* the temporal Mamba stack.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalADRiverDynamics(nn.Module):
    """
    One Euler step along time: neighbors are previous / next frame at the same (n).

    A: softmax weights from alignment between predicted flow direction and temporal edge vectors.
    D: graph Laplacian on the length-L chain 0.5*(h_{t-1}+h_{t+1}) - h_t, gated by diffusion_head(h).
    R: MLP on [h, flow, temporal span statistics].
    """

    def __init__(
        self,
        c_dim: int,
        advection_tau: float = 0.15,
        learnable_dt: bool = True,
        dt_init: float = 0.5,
    ):
        super().__init__()
        self.c_dim = c_dim
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
        r_in = c_dim + 3 + 2
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
        return self.log_dt.exp().clamp(1e-5, 5.0)

    def forward(
        self,
        f: torch.Tensor,
        xyz: torch.Tensor,
        uncertainty_prior: torch.Tensor | None = None,
        return_components: bool = False,
    ):
        """
        f: [B, L, N, C], xyz: [B, L, N, 3]
        """
        bsz, l, n, c = f.shape
        if l < 2:
            if return_components:
                return f, {"skipped": True, "reason": "L<2"}
            return f

        # Flow prediction: Conv1d over the time axis — reshape to (B*N, C, L)
        fn = f.permute(0, 2, 1, 3).reshape(bsz * n, l, c)
        hb = fn.transpose(1, 2)
        flow_pred = self.flow_head(hb).transpose(1, 2).reshape(bsz, n, l, 3).permute(0, 2, 1, 3).contiguous()

        diff_sc = F.softplus(self.diffusion_head(hb)).transpose(1, 2)
        diffusion = diff_sc.reshape(bsz, n, l, 1).permute(0, 2, 1, 3).contiguous()
        uncert_sc = torch.sigmoid(self.uncertainty_head(hb)).transpose(1, 2)
        uncertainty = uncert_sc.reshape(bsz, n, l, 1).permute(0, 2, 1, 3).contiguous()
        if uncertainty_prior is not None:
            uncertainty_prior = uncertainty_prior.clamp(0.0, 1.0)
            uncertainty = 0.5 * (uncertainty + uncertainty_prior)

        f_rp = torch.cat([f[:, :1], f, f[:, -1:]], dim=1)
        xyz_rp = torch.cat([xyz[:, :1], xyz, xyz[:, -1:]], dim=1)
        f_prev = f_rp[:, :-2]
        f_next = f_rp[:, 2:]
        xyz_prev = xyz_rp[:, :-2]
        xyz_next = xyz_rp[:, 2:]

        lap_t = 0.5 * (f_prev + f_next) - f
        diffusion_eff = diffusion * (1.0 + uncertainty)
        diff_term = diffusion_eff * lap_t

        center_xyz = xyz
        delta_stack = torch.stack([xyz_prev - center_xyz, xyz_next - center_xyz], dim=3)
        f_stack = torch.stack([f_prev, f_next], dim=3)
        v_dir = F.normalize(flow_pred, dim=-1, eps=1e-6)
        d_norm = F.normalize(delta_stack, dim=-1, eps=1e-6)
        cos_align = (d_norm * v_dir.unsqueeze(3)).sum(dim=-1)
        w = F.softmax(cos_align / self.advection_tau, dim=3)
        h_flow = (w.unsqueeze(-1) * f_stack).sum(dim=3)
        adv_feat = h_flow - f
        flow_global = flow_pred.mean(dim=2)  # [B,L,3]
        flow_global_feat = self.global_flow_proj(flow_global).unsqueeze(2)  # [B,L,1,C]
        adv_gate = self.global_adv_gate(hb).transpose(1, 2).reshape(bsz, n, l, c).permute(0, 2, 1, 3)
        adv_global = adv_gate * flow_global_feat
        adv_feat = adv_feat + adv_global

        edge_len = delta_stack.norm(dim=-1)
        geom_t = torch.cat(
            [edge_len.min(dim=-1, keepdim=True)[0], edge_len.max(dim=-1, keepdim=True)[0]],
            dim=-1,
        )

        r_cat = torch.cat([f, flow_pred, geom_t], dim=-1)
        d_r = r_cat.shape[-1]
        rn = r_cat.permute(0, 2, 1, 3).reshape(bsz * n, l, d_r).transpose(1, 2)
        reac_term = self.reaction_mlp(rn).reshape(bsz, n, c, l).permute(0, 3, 1, 2).contiguous()

        dt = self.dt()
        f_next_step = f + dt * (adv_feat + diff_term + reac_term)

        comp = {
            "flow_pred_t": flow_pred,
            "adv_feat_t": adv_feat,
            "lap_t": lap_t,
            "diffusion_t": diffusion,
            "diffusion_eff_t": diffusion_eff,
            "uncertainty_t": uncertainty,
            "diff_term_t": diff_term,
            "reaction_t": reac_term,
            "advection_weights_t": w,
            "adv_global_t": adv_global,
            "dt": dt,
        }
        if return_components:
            return f_next_step, comp
        return f_next_step


class TemporalADRiverRefinement(nn.Module):
    """Unrolled temporal ADR steps (weight-sharing)."""

    def __init__(self, dynamics: TemporalADRiverDynamics, num_iters: int = 1):
        super().__init__()
        self.dynamics = dynamics
        self.num_iters = num_iters

    def forward(
        self,
        f: torch.Tensor,
        xyz: torch.Tensor,
        uncertainty_prior: torch.Tensor | None = None,
    ):
        aux_all = []
        h = f
        for _ in range(self.num_iters):
            h, comp = self.dynamics(
                h,
                xyz,
                uncertainty_prior=uncertainty_prior,
                return_components=True,
            )
            aux_all.append(comp)
        return h, aux_all
