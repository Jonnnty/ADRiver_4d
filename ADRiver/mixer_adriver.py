"""
Temporal Mamba stack with **in-block flow-conditioned reaction** (local R after each SSM layer).
Duplicates the upstream MixerModel layout so we do not modify vendored `mamba.py`.
"""

from __future__ import annotations

import sys
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn

# Resolve colocated backbone modules (same rule as `encoder.py`)
_ROOT = Path(__file__).resolve().parent
_MOD = _ROOT / "modules"
if str(_MOD) not in sys.path:
    sys.path.insert(0, str(_MOD))

from timm.models.layers import DropPath  # noqa: E402

from mamba import _init_weights, create_block_flow  # noqa: E402

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class FlowConditionedReaction(nn.Module):
    """
    R(h, v): local reaction as residual MLP on [token, scene-flow at token].
    Initialized near zero on the last linear so training starts as vanilla Mamba.
    """

    def __init__(self, d_model: int, hidden_ratio: int = 4):
        super().__init__()
        hid = max(d_model * 2 // max(hidden_ratio, 1), 64)
        self.net = nn.Sequential(
            nn.Linear(d_model + 3, hid),
            nn.ReLU(inplace=True),
            nn.Linear(hid, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(0.25))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, h: torch.Tensor, flow_seq: torch.Tensor) -> torch.Tensor:
        x = torch.cat([h, flow_seq], dim=-1)
        return (self.scale * self.net(x)).to(dtype=h.dtype)


class IterativeFlowPropagation(nn.Module):
    """
    Flow-guided global-to-local propagation unit (iterative).
    This approximates advection-style global motion propagation inside token space.
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.global_proj = nn.Sequential(
            nn.Linear(3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.local_gate = nn.Sequential(
            nn.Linear(d_model + 3, d_model),
            nn.Sigmoid(),
        )
        self.delta_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )
        self.scale = nn.Parameter(torch.tensor(0.2))

    def forward(self, h: torch.Tensor, flow_seq: torch.Tensor) -> torch.Tensor:
        # h, flow_seq: [B,S,*]
        global_flow = flow_seq.mean(dim=1)  # [B,3]
        global_token = self.global_proj(global_flow).unsqueeze(1)  # [B,1,D]
        gate = self.local_gate(torch.cat([h, flow_seq], dim=-1))  # [B,S,D]
        delta = gate * (global_token - h)
        delta = self.delta_proj(delta)
        return h + self.scale * delta


class MixerModelFlowReaction(nn.Module):
    """
    Same API as `intra_mamba.MixerModel` plus required `flow_flat [B,S,3]` (frame-major S=L*N).
    After each SSM block: h <- h + R(h, flow).
    """

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        ssm_cfg=None,
        norm_epsilon: float = 1e-5,
        rms_norm: bool = False,
        initializer_cfg=None,
        fused_add_norm: bool = False,
        residual_in_fp32: bool = False,
        drop_out_in_block: float = 0.0,
        drop_path: float = 0.1,
        drop_path_rate: float = 0.1,
        bimamba: bool = True,
        device=None,
        dtype=None,
        reaction_hidden_ratio: int = 4,
        propagation_steps: int = 2,
    ) -> None:
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        if self.fused_add_norm:
            if layer_norm_fn is None or rms_norm_fn is None:
                raise ImportError("Triton LayerNorm / RMSNorm kernels required when fused_add_norm=True")

        self.layers = nn.ModuleList(
            [
                create_block_flow(
                    d_model,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=drop_path,
                    bimamba=bimamba,
                    flow_cond=True,
                    **factory_kwargs,
                )
                for i in range(n_layer)
            ]
        )

        self.flow_reactions = nn.ModuleList(
            [FlowConditionedReaction(d_model, hidden_ratio=reaction_hidden_ratio) for _ in range(n_layer)]
        )
        self.flow_propagations = nn.ModuleList([IterativeFlowPropagation(d_model) for _ in range(n_layer)])
        self.propagation_steps = propagation_steps

        self.norm_f = (nn.LayerNorm if not rms_norm else RMSNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                **(initializer_cfg if initializer_cfg is not None else {}),
            )
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.drop_out_in_block = nn.Dropout(drop_out_in_block) if drop_out_in_block > 0.0 else nn.Identity()

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return {
            i: layer.allocate_inference_cache(batch_size, max_seqlen, dtype=dtype, **kwargs)
            for i, layer in enumerate(self.layers)
        }

    def forward(self, embedding_tokens, flow_flat: torch.Tensor, pos=None, inference_params=None):
        hidden_states = embedding_tokens
        residual = None
        if pos is not None:
            hidden_states = hidden_states + pos

        for i, layer in enumerate(self.layers):
            hidden_states, residual = layer(
                hidden_states,
                residual,
                inference_params=inference_params,
                flow_features=flow_flat,
            )
            hidden_states = self.drop_out_in_block(hidden_states)
            hidden_states = hidden_states + self.flow_reactions[i](hidden_states, flow_flat)
            for _ in range(self.propagation_steps):
                hidden_states = self.flow_propagations[i](hidden_states, flow_flat)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
            hidden_states = fused_add_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
            )

        return hidden_states


def frame_major_flow(xyz_fine: torch.Tensor) -> torch.Tensor:
    """
    xyz_fine: [B, L, N, 3] -> flow_seq: [B, L*N, 3] finite-difference along time (replicate last).
    """
    bsz, l, n, _ = xyz_fine.shape
    flow = torch.zeros(bsz, l, n, 3, device=xyz_fine.device, dtype=xyz_fine.dtype)
    if l >= 2:
        flow[:, :-1] = xyz_fine[:, 1:] - xyz_fine[:, :-1]
        flow[:, -1] = flow[:, -2]
    return flow.reshape(bsz, l * n, 3)
