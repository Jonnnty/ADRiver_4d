#!/usr/bin/env python3
# Copyright (c) 2026
"""
Train ``ADRiver4DEncoder`` on ``adriver_train.npz`` (Depth Anything → ADRiver export).

Uses supervised / consistency signals from the encoder forward pass:
  - flow supervision (smooth L1 vs finite-difference flow of xyz),
  - optional multi-scale flow consistency (mid vs fine),
  - optional uncertainty consistency (prior vs predicted uncertainty).

Outputs:
  - checkpoints (``latest.pt``, optional ``best.pt``),
  - ``train_log.jsonl`` (one JSON object per epoch),
  - ``heatmaps/`` — time×point index heatmaps (no 3-D scatter),
  - ``loss_curve.png`` — total loss vs epoch.

Example (from the repository root; set paths to match your machine):

  PYTHONPATH=$PWD:$PWD/ADRiver/modules \\
  python scripts/train_adriver_from_npz.py \\
    --npz ./data/adriver_train.npz \\
    --out-dir ./runs/my_experiment \\
    --epochs 50 \\
    --max-t 16 --max-n 2048
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch


def _grad_scaler(enabled: bool, device: torch.device):
    dt = device.type
    if dt == "cuda":
        return torch.amp.GradScaler(dt, enabled=enabled)
    return torch.amp.GradScaler("cpu", enabled=False)


def _autocast(device: torch.device, enabled: bool):
    return torch.amp.autocast(device_type=device.type, enabled=enabled and device.type == "cuda")

# Repo root (parent of ``scripts/``)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_MOD = ROOT / "ADRiver" / "modules"
if str(_MOD) not in sys.path:
    sys.path.insert(0, str(_MOD))


def _import_encoder():
    from ADRiver.encoder import ADRiver4DEncoder

    return ADRiver4DEncoder


def load_npz_clip(
    path: Path,
    device: torch.device,
    max_t: int,
    max_n: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, dict]:
    data = np.load(path)
    xyzs = torch.from_numpy(np.asarray(data["xyzs"], dtype=np.float32)).to(device)
    rgbs = torch.from_numpy(np.asarray(data["rgbs"], dtype=np.float32)).to(device)
    unc: torch.Tensor | None = None
    if "uncertainty_prior" in data.files:
        unc = torch.from_numpy(np.asarray(data["uncertainty_prior"], dtype=np.float32)).to(device)

    b, t, n, _ = xyzs.shape
    meta = {"T_orig": int(t), "N_orig": int(n), "B": int(b)}
    if max_t > 0 and t > max_t:
        xyzs = xyzs[:, :max_t]
        rgbs = rgbs[:, :max_t]
        if unc is not None:
            unc = unc[:, :max_t]
        t = max_t
    if max_n > 0 and n > max_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(n, size=max_n, replace=False)
        idx_t = torch.from_numpy(idx).long().to(device)
        xyzs = torch.gather(xyzs, 2, idx_t.view(1, 1, -1, 1).expand(b, t, max_n, 3))
        rgbs = torch.gather(rgbs, 2, idx_t.view(1, 1, -1, 1).expand(b, t, max_n, 3))
        if unc is not None:
            unc = torch.gather(unc, 2, idx_t.view(1, 1, -1, 1).expand(b, t, max_n, 1))
        n = max_n
    meta.update({"T": int(t), "N": int(n)})
    return xyzs, rgbs, unc, meta


def build_loss(aux: dict, weights: dict) -> tuple[torch.Tensor, dict[str, float]]:
    """Combine available auxiliary losses; missing keys are skipped."""
    parts: list[tuple[str, torch.Tensor]] = []
    if "loss_flow_supervision" in aux and weights.get("flow", 0.0) > 0:
        parts.append(("loss_flow_supervision", weights["flow"] * aux["loss_flow_supervision"]))
    if "loss_flow_multiscale" in aux and weights.get("multi", 0.0) > 0:
        parts.append(("loss_flow_multiscale", weights["multi"] * aux["loss_flow_multiscale"]))
    if "loss_uncertainty_consistency" in aux and weights.get("unc", 0.0) > 0:
        parts.append(
            ("loss_uncertainty_consistency", weights["unc"] * aux["loss_uncertainty_consistency"])
        )

    if not parts:
        raise RuntimeError(
            "No trainable loss terms (need T>=2 for flow supervision and/or mid-scale flow / uncertainty). "
            "Increase --max-t to at least 2 or enable mid-scale ADR / provide uncertainty_prior."
        )

    total = sum(p[1] for p in parts)
    log = {name: float(v.detach().item()) for name, v in parts}
    log["total"] = float(total.detach().item())
    return total, log


@torch.no_grad()
def export_heatmaps(
    enc: torch.nn.Module,
    xyzs: torch.Tensor,
    rgbs: torch.Tensor,
    unc: torch.Tensor | None,
    out_dir: Path,
    device: torch.device,
) -> None:
    """Save 2-D heatmaps: time × point index. No 3-D scatter."""
    enc.eval()
    out_dir.mkdir(parents=True, exist_ok=True)
    z, xyz_o, aux = enc(xyzs, rgbs, uncertainty_prior=unc, return_aux=True)
    # [B,L,N,C] -> latent L2 norm per (t, n)
    zn = torch.linalg.norm(z[0], dim=-1).float().cpu().numpy()
    _save_heatmap(zn, out_dir / "heatmap_latent_l2norm_TxN.png", "‖z‖_2 (T × N')", cmap="magma")

    if aux and "adr_fine_last" in aux:
        fp = aux["adr_fine_last"]["flow_pred"][0]  # [L, N, 3]
        fm = torch.linalg.norm(fp, dim=-1).float().cpu().numpy()
        _save_heatmap(fm, out_dir / "heatmap_adr_fine_flow_norm_TxN.png", "ADR fine ‖flow‖ (T × N')", cmap="inferno")

    # Optional: supervised flow vs target magnitude (if L>=2)
    if aux and "flow_target" in aux and aux["flow_target"].shape[1] > 0:
        tgt = aux["flow_target"][0]
        tm = torch.linalg.norm(tgt, dim=-1).float().cpu().numpy()
        _save_heatmap(tm, out_dir / "heatmap_flow_target_norm_Tminus1xN.png", "Δxyz target ‖·‖", cmap="viridis")

    # XYZ spatial spread per frame (scalar): std over points for quick dynamics cue
    sp = xyz_o[0].float().std(dim=1).mean(dim=-1).cpu().numpy()  # [L]
    _save_1d_bar(sp, out_dir / "plot_xyz_std_per_frame.png", "mean std(xyz) per frame")


def _save_heatmap(mat: np.ndarray, path: Path, title: str, cmap: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(8, mat.shape[1] / 64), max(4, mat.shape[0] / 8)))
    im = ax.imshow(mat, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_xlabel("point index n′")
    ax.set_ylabel("frame t")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _save_1d_bar(values: np.ndarray, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.bar(np.arange(len(values)), values, color="steelblue")
    ax.set_xlabel("frame")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_loss_curve(log_path: Path, out_png: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = []
    totals = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        epochs.append(row["epoch"])
        totals.append(row["loss"]["total"])
    if not epochs:
        return
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(epochs, totals, "-o", ms=3, lw=1.5, color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("training loss")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    ap = argparse.ArgumentParser("ADRiver-4D training from adriver_train.npz")
    ap.add_argument("--npz", type=Path, required=True, help="Path to adriver_train.npz")
    ap.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Run directory (checkpoints, train_log.jsonl, heatmaps, loss_curve.png).",
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-5)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--warmup-epochs", type=int, default=3)
    ap.add_argument("--max-t", type=int, default=0, help="0 = full T in npz")
    ap.add_argument("--max-n", type=int, default=0, help="0 = full N")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--amp", action="store_true", help="Automatic mixed precision")
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--w-flow", type=float, default=1.0)
    ap.add_argument(
        "--w-multi",
        type=float,
        default=0.05,
        help="Weight for multi-scale flow MSE (often large magnitude vs flow supervision).",
    )
    ap.add_argument("--w-unc", type=float, default=0.25)
    ap.add_argument("--resume", type=Path, default=None, help="Resume from checkpoint .pt")
    ap.add_argument("--save-every", type=int, default=5)
    ap.add_argument("--heatmaps-every", type=int, default=10, help="Export heatmaps every N epochs (0=only last)")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    set_seed(args.seed)
    out: Path = args.out_dir.expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)
    heatmap_root = out / "heatmaps"
    log_path = out / "train_log.jsonl"

    ADRiver4DEncoder = _import_encoder()
    enc = ADRiver4DEncoder(
        use_mid_scale_adr=True,
        adr_fine_iters=3,
        use_mamba_flow_reaction=True,
        use_multiscale_flow_consistency=True,
    ).to(device)

    opt = torch.optim.AdamW(enc.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = _grad_scaler(args.amp, device)

    start_epoch = 0
    best_loss = float("inf")
    if args.resume is not None and args.resume.is_file():
        ck = torch.load(args.resume, map_location=device)
        enc.load_state_dict(ck["model"])
        if "optimizer" in ck:
            opt.load_state_dict(ck["optimizer"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck.get("epoch", -1)) + 1
        best_loss = float(ck.get("best_loss", float("inf")))
        print(f"[train] resumed from {args.resume} at epoch {start_epoch}")

    xyzs, rgbs, unc, meta = load_npz_clip(args.npz, device, args.max_t, args.max_n, args.seed)
    if meta["T"] < 2:
        sys.stderr.write(
            "[train] WARNING: T<2 — flow supervision is disabled; ensure multi-scale or uncertainty loss applies.\n"
        )

    loss_w = {"flow": args.w_flow, "multi": args.w_multi, "unc": args.w_unc}

    def lr_at(epoch: int) -> float:
        if epoch < args.warmup_epochs:
            return args.lr * float(epoch + 1) / float(max(1, args.warmup_epochs))
        # cosine decay to 0.1 * lr
        t = (epoch - args.warmup_epochs) / float(max(1, args.epochs - args.warmup_epochs))
        return float(0.1 * args.lr + 0.9 * args.lr * 0.5 * (1.0 + math.cos(math.pi * min(1.0, t))))

    for epoch in range(start_epoch, args.epochs):
        lr = lr_at(epoch)
        for g in opt.param_groups:
            g["lr"] = lr

        enc.train()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)

        with _autocast(device, args.amp):
            z, xyz_o, aux = enc(xyzs, rgbs, uncertainty_prior=unc, return_aux=True)
            loss, log = build_loss(aux, loss_w)

        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(enc.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()

        row = {
            "epoch": epoch,
            "time_s": time.time() - t0,
            "lr": lr,
            "meta": meta,
            "loss": log,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(f"epoch {epoch+1}/{args.epochs}  total={log['total']:.6f}  {log}  lr={lr:.2e}")

        is_best = log["total"] < best_loss
        if is_best:
            best_loss = log["total"]

        if (epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs:
            ck = {
                "epoch": epoch,
                "model": enc.state_dict(),
                "optimizer": opt.state_dict(),
                "scaler": scaler.state_dict(),
                "best_loss": best_loss,
                "meta": meta,
                "args": vars(args),
            }
            torch.save(ck, ckpt_dir / "latest.pt")
            if is_best:
                torch.save(ck, ckpt_dir / "best.pt")

        he = args.heatmaps_every
        do_hm = he > 0 and (epoch + 1) % he == 0
        if do_hm or epoch + 1 == args.epochs:
            export_heatmaps(enc, xyzs, rgbs, unc, heatmap_root / f"epoch_{epoch+1:04d}", device)

    plot_loss_curve(log_path, out / "loss_curve.png")
    export_heatmaps(enc, xyzs, rgbs, unc, heatmap_root / "final", device)
    print(f"[train] done. Checkpoints: {ckpt_dir}  logs: {log_path}  heatmaps: {heatmap_root}")


if __name__ == "__main__":
    main()
