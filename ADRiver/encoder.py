"""
ADRiver-4D Encoder (encoding only): P4D pyramid + temporal ADR + Mamba (+ in-layer R) + spatial ADR.

• Spatial ADR @ 512 (mid) and @ 1024 (fine), same kNN semantics as before.
• Temporal ADR along L before and after Mamba (shared A/D/R *form*, chain neighbors in time).
• Mamba: optional flow-conditioned reaction after each SSM block.
• Optional coarse→fine consistency on predicted scene-flow heads (512 upsampled vs 1024).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# In-repo backbone modules (P4D + Mamba stack) — now colocated under ADRiver/
_PKG_ROOT = Path(__file__).resolve().parent
_MAMBA_MODULES = _PKG_ROOT / "modules"
if str(_MAMBA_MODULES) not in sys.path:
    sys.path.insert(0, str(_MAMBA_MODULES))

from intra_mamba import MixerModel  # noqa: E402
from point_4d_convolution import P4DConv  # noqa: E402

from .adr_operator import ADRiverDynamics, ADRiverRefinement
from .geometry import soft_knn_interpolate_vectors
from .mixer_adriver import MixerModelFlowReaction, frame_major_flow
from .temporal_dynamics import TemporalADRiverDynamics, TemporalADRiverRefinement


class ADRiver4DEncoder(nn.Module):
    """
    Full stack:
      P4D → (spatial ADR mid C=512) → P4D → temporal ADR (pre) → Mamba ± flow-R → temporal ADR (post)
      → spatial ADR fine (iterative) + optional multi-scale flow consistency readout in aux.
    """

    def __init__(
        self,
        radius: float = 0.9,
        nsamples: int = 9,
        d_model: int = 1024,
        mamba_layers: int = 12,
        use_mid_scale_adr: bool = True,
        mid_k_neighbors: int = 12,
        mid_advection_tau: float = 0.2,
        fine_k_neighbors: int = 16,
        fine_advection_tau: float = 0.15,
        adr_fine_iters: int = 3,
        drop_out_in_block: float = 0.1,
        drop_path: float = 0.1,
        # --- temporal ADR (chain along L; shares tau semantics with fine by default) ---
        use_temporal_adr_pre: bool = True,
        temporal_pre_iters: int = 1,
        use_temporal_adr_post: bool = True,
        temporal_post_iters: int = 1,
        temporal_advection_tau: float | None = None,
        temporal_dt_init_pre: float = 0.45,
        temporal_dt_init_post: float = 0.35,
        # --- Mamba: flow-conditioned reaction inside stack ---
        use_mamba_flow_reaction: bool = True,
        mamba_propagation_steps: int = 2,
        use_flow_reorder: bool = True,
        # --- coarse / fine flow consistency (aux tensor for training) ---
        use_multiscale_flow_consistency: bool = True,
        multiscale_knn_k: int = 3,
        multiscale_interp_sigma: float = 0.12,
        flow_supervision_refine_iters: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        tau_t = temporal_advection_tau if temporal_advection_tau is not None else fine_advection_tau

        self.use_temporal_adr_pre = use_temporal_adr_pre
        self.use_temporal_adr_post = use_temporal_adr_post
        self.use_mamba_flow_reaction = use_mamba_flow_reaction
        self.use_flow_reorder = use_flow_reorder
        self.use_multiscale_flow_consistency = use_multiscale_flow_consistency
        self.multiscale_knn_k = multiscale_knn_k
        self.multiscale_interp_sigma = multiscale_interp_sigma

        self.conv1 = P4DConv(
            in_planes=3,
            mlp_planes=[32, 64, 128],
            mlp_batch_norm=[True, True, True],
            mlp_activation=[True, True, True],
            spatial_kernel_size=[radius, nsamples],
            temporal_kernel_size=1,
            spatial_stride=4,
            temporal_stride=1,
            temporal_padding=[0, 0],
        )
        self.conv2 = P4DConv(
            in_planes=128,
            mlp_planes=[128, 128, 256],
            mlp_batch_norm=[True, True, True],
            mlp_activation=[True, True, True],
            spatial_kernel_size=[2 * radius, nsamples],
            temporal_kernel_size=1,
            spatial_stride=4,
            temporal_stride=1,
            temporal_padding=[0, 0],
        )
        self.conv3 = P4DConv(
            in_planes=256,
            mlp_planes=[256, 256, 512],
            mlp_batch_norm=[True, True, True],
            mlp_activation=[True, True, True],
            spatial_kernel_size=[2 * 2 * radius, nsamples],
            temporal_kernel_size=3,
            spatial_stride=4,
            temporal_stride=1,
            temporal_padding=[1, 1],
        )

        self.use_mid_scale_adr = use_mid_scale_adr
        self.adr_mid = None
        if use_mid_scale_adr:
            self.adr_mid = ADRiverDynamics(
                c_dim=512,
                k_neighbors=mid_k_neighbors,
                advection_tau=mid_advection_tau,
                learnable_dt=True,
                dt_init=0.8,
            )

        self.conv4 = P4DConv(
            in_planes=512,
            mlp_planes=[512, 512, d_model],
            mlp_batch_norm=[True, True, True],
            mlp_activation=[True, True, True],
            spatial_kernel_size=[2 * 2 * 2 * radius, nsamples],
            temporal_kernel_size=1,
            spatial_stride=2,
            temporal_stride=1,
            temporal_padding=[0, 0],
        )

        self.emb_relu = nn.ReLU()

        if use_mamba_flow_reaction:
            self.mamba_blocks = MixerModelFlowReaction(
                d_model=d_model,
                n_layer=mamba_layers,
                rms_norm=False,
                drop_out_in_block=drop_out_in_block,
                drop_path=drop_path,
                propagation_steps=mamba_propagation_steps,
            )
        else:
            self.mamba_blocks = MixerModel(
                d_model=d_model,
                n_layer=mamba_layers,
                rms_norm=False,
                drop_out_in_block=drop_out_in_block,
                drop_path=drop_path,
            )

        self.temporal_pre = None
        if use_temporal_adr_pre:
            self.temporal_pre = TemporalADRiverRefinement(
                TemporalADRiverDynamics(
                    c_dim=d_model,
                    advection_tau=tau_t,
                    learnable_dt=True,
                    dt_init=temporal_dt_init_pre,
                ),
                num_iters=temporal_pre_iters,
            )

        self.temporal_post = None
        if use_temporal_adr_post:
            self.temporal_post = TemporalADRiverRefinement(
                TemporalADRiverDynamics(
                    c_dim=d_model,
                    advection_tau=tau_t,
                    learnable_dt=True,
                    dt_init=temporal_dt_init_post,
                ),
                num_iters=temporal_post_iters,
            )

        self.adr_fine = ADRiverRefinement(
            ADRiverDynamics(
                c_dim=d_model,
                k_neighbors=fine_k_neighbors,
                advection_tau=fine_advection_tau,
                learnable_dt=True,
                dt_init=1.0,
            ),
            num_iters=adr_fine_iters,
        )
        self.flow_supervision_head = nn.Conv1d(d_model, 3, kernel_size=1)
        self.flow_supervision_refine = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(d_model + 3, d_model, kernel_size=1),
                    nn.ReLU(inplace=True),
                    nn.Conv1d(d_model, 3, kernel_size=1),
                )
                for _ in range(flow_supervision_refine_iters)
            ]
        )

    @staticmethod
    def _argsort_gather(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        return torch.gather(x, 2, idx.unsqueeze(-1).expand(-1, -1, -1, x.shape[-1]))

    @staticmethod
    def _invert_permutation(idx: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(idx)
        base = torch.arange(idx.shape[-1], device=idx.device).view(1, 1, -1).expand_as(idx)
        inv.scatter_(2, idx, base)
        return inv

    def forward(
        self,
        xyzs: torch.Tensor,
        rgbs: torch.Tensor,
        uncertainty_prior: torch.Tensor | None = None,
        return_aux: bool = False,
    ):
        """
        xyzs: [B, T, N, 3]
        rgbs: [B, T, N, 3]

        Returns:
          z: [B, L, N', C] encoded tokens after ADRiver refinement
          xyz_out: [B, L, N', 3] downsampled coordinates (fine scale)
          aux: optional dict; may include `loss_flow_multiscale` for training when enabled.

        Note:
          ``P4DConv`` expects ``features`` shaped ``(B, T, C, N)``. Inputs shaped ``(B, T, N, C)``
          (e.g. RGB as last dim) are accepted and transposed here.
        """
        # P4DConv uses (B, T, C, N). (B, T, N, C) has last dim 3 and penultimate dim N≠3.
        if rgbs.dim() == 4 and rgbs.size(-1) == 3 and rgbs.size(-2) != 3:
            rgbs = rgbs.permute(0, 1, 3, 2).contiguous()
        new_xyzs1, new_features1 = self.conv1(xyzs, rgbs)
        new_xyzs2, new_features2 = self.conv2(new_xyzs1, new_features1)
        new_xyzs3, new_features3 = self.conv3(new_xyzs2, new_features2)

        flow_mid = None
        aux_mid = None
        if self.adr_mid is not None:
            # [B,L,512,N] -> [B,L,N,512]
            f3 = new_features3.permute(0, 1, 3, 2).contiguous()
            uncertainty_mid = None
            if uncertainty_prior is not None:
                uncertainty_mid = soft_knn_interpolate_vectors(
                    xyzs,
                    uncertainty_prior.expand(-1, -1, -1, 3),
                    new_xyzs3,
                    k=self.multiscale_knn_k,
                    sigma=self.multiscale_interp_sigma,
                )[..., :1]
            f3, aux_mid = self.adr_mid(
                f3,
                new_xyzs3,
                uncertainty_prior=uncertainty_mid,
                return_components=True,
            )
            new_features3 = f3.permute(0, 1, 3, 2).contiguous()
            flow_mid = aux_mid["flow_pred"]

        new_xyzs4, new_features4 = self.conv4(new_xyzs3, new_features3)

        bsz, l, _, n = new_features4.size()
        features = new_features4.permute(0, 1, 3, 2).contiguous()
        # [B,L,N,C]
        xyz_reordered = new_xyzs4
        flow_reorder_idx = None
        if self.use_flow_reorder:
            flow_seed = torch.zeros_like(new_xyzs4)
            if l >= 2:
                flow_seed[:, :-1] = new_xyzs4[:, 1:] - new_xyzs4[:, :-1]
                flow_seed[:, -1] = flow_seed[:, -2]
            global_axis = F.normalize(flow_seed.mean(dim=2), dim=-1, eps=1e-6)
            centered = new_xyzs4 - new_xyzs4.mean(dim=2, keepdim=True)
            score = (centered * global_axis.unsqueeze(2)).sum(dim=-1)
            flow_reorder_idx = torch.argsort(score, dim=2)
            features = self._argsort_gather(features, flow_reorder_idx)
            xyz_reordered = self._argsort_gather(new_xyzs4, flow_reorder_idx)

        uncertainty_fine = None
        if uncertainty_prior is not None:
            uncertainty_fine = soft_knn_interpolate_vectors(
                xyzs,
                uncertainty_prior.expand(-1, -1, -1, 3),
                xyz_reordered,
                k=self.multiscale_knn_k,
                sigma=self.multiscale_interp_sigma,
            )[..., :1]

        aux_temporal_pre = None
        if self.temporal_pre is not None:
            features, aux_temporal_pre = self.temporal_pre(
                features,
                xyz_reordered,
                uncertainty_prior=uncertainty_fine,
            )

        features_flat = torch.reshape(features, (bsz, l * n, features.shape[3]))
        embedding = self.emb_relu(features_flat)
        flow_flat = frame_major_flow(xyz_reordered)

        if self.use_mamba_flow_reaction:
            features_flat = self.mamba_blocks(embedding, flow_flat)
        else:
            features_flat = self.mamba_blocks(embedding)

        features = torch.reshape(features_flat, (bsz, l, n, features_flat.shape[2]))
        features = features.contiguous()

        aux_temporal_post = None
        if self.temporal_post is not None:
            features, aux_temporal_post = self.temporal_post(
                features,
                xyz_reordered,
                uncertainty_prior=uncertainty_fine,
            )

        flow_feat = features.permute(0, 1, 3, 2).reshape(bsz * l, self.d_model, n)
        flow_pred_sup = self.flow_supervision_head(flow_feat)
        for refine in self.flow_supervision_refine:
            delta = refine(torch.cat([flow_feat, flow_pred_sup], dim=1))
            flow_pred_sup = flow_pred_sup + delta
        flow_pred_sup = flow_pred_sup.reshape(bsz, l, 3, n).permute(0, 1, 3, 2).contiguous()

        # ``ADRiverDynamics`` / ``ADRiverRefinement`` expect ``[B, L, N, C]`` (same as ``features`` here).
        # Do *not* permute to ``[B, L, C, N]`` — that swaps N/C and breaks Conv1d heads (e.g. 32 vs 1024).
        f_seq = features.contiguous()
        f_refined, adr_fine_list = self.adr_fine(
            f_seq,
            xyz_reordered,
            uncertainty_prior=uncertainty_fine,
        )

        z = f_refined
        xyz_out = xyz_reordered
        if flow_reorder_idx is not None:
            inv_idx = self._invert_permutation(flow_reorder_idx)
            z = self._argsort_gather(z, inv_idx)
            xyz_out = self._argsort_gather(xyz_out, inv_idx)
            flow_pred_sup = self._argsort_gather(flow_pred_sup, inv_idx)

        if not return_aux:
            return z, xyz_out

        aux = {
            "adr_fine_last": adr_fine_list[-1],
            "adr_fine_all": adr_fine_list,
            "flow_target": xyz_out[:, 1:, :, :] - xyz_out[:, :-1, :, :],
            "flow_supervised_pred": flow_pred_sup,
            "temporal_pre": aux_temporal_pre,
            "temporal_post": aux_temporal_post,
        }
        if l >= 2:
            aux["loss_flow_supervision"] = F.smooth_l1_loss(
                flow_pred_sup[:, :-1], aux["flow_target"]
            )
        if aux_mid is not None:
            aux["adr_mid"] = aux_mid
        if uncertainty_prior is not None:
            aux["uncertainty_prior"] = uncertainty_prior

        if (
            self.use_multiscale_flow_consistency
            and flow_mid is not None
            and l >= 1
            and new_xyzs3.shape[2] > 0
        ):
            flow_up = soft_knn_interpolate_vectors(
                new_xyzs3,
                flow_mid,
                xyz_reordered,
                k=self.multiscale_knn_k,
                sigma=self.multiscale_interp_sigma,
            )
            flow_fine = adr_fine_list[-1]["flow_pred"]
            aux["flow_mid_upsampled"] = flow_up
            aux["flow_fine_last"] = flow_fine
            aux["loss_flow_multiscale"] = F.mse_loss(flow_up, flow_fine)
        if uncertainty_prior is not None:
            uncertainty_fine_last = adr_fine_list[-1]["uncertainty"]
            uncertainty_proj = soft_knn_interpolate_vectors(
                xyzs,
                uncertainty_prior.expand(-1, -1, -1, 3),
                xyz_reordered,
                k=self.multiscale_knn_k,
                sigma=self.multiscale_interp_sigma,
            )[..., :1]
            aux["uncertainty_fine_last"] = uncertainty_fine_last
            aux["uncertainty_fine_prior"] = uncertainty_proj
            aux["loss_uncertainty_consistency"] = F.l1_loss(uncertainty_fine_last, uncertainty_proj)

        return z, xyz_out, aux
