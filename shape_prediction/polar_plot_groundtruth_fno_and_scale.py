#!/usr/bin/env python3
"""
Inspeção do modelo FNO — apenas forma normalizada:
  E(f, theta) -> S_tilde(f, theta) [FNO]

Não utiliza o modelo de amplitude. A predição e o ground truth
são comparados na escala normalizada [-1, 1].
"""

import argparse
import os
import numpy as np
import h5py
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.gridspec import GridSpec

from fno_diffusion.model import make_fno_2d

_trapz = getattr(np, "trapezoid", None) or np.trapz


# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

class FNOOnly(nn.Module):
    def __init__(self, n_modes=(16, 16), hidden_channels=64):
        super().__init__()
        self.field_model = make_fno_2d(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=1,
            out_channels=1,
        )

    def forward(self, x):
        return self.field_model(x)


def clean_state_dict(state):
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise ValueError("Checkpoint inválido: não contém state_dict.")
    cleaned = {}
    for k, v in state.items():
        if k == "_metadata":
            continue
        if k.startswith("module."):
            k = k[len("module."):]
        if not k.startswith("field_model."):
            k = "field_model." + k
        cleaned[k] = v
    return cleaned


def load_model(model_path, n_modes=(16, 16), hidden_channels=64):
    model = FNOOnly(n_modes=n_modes, hidden_channels=hidden_channels)
    state = torch.load(model_path, map_location="cpu", weights_only=False)
    state = clean_state_dict(state)
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f" Modelo FNO carregado: {model_path}")
    return model


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def rel_l2(pred, target):
    return float(np.linalg.norm((pred - target).ravel()) / (np.linalg.norm(target.ravel()) + 1e-30))


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def _build_title(idx, fp, Hs, gamma, th0, s):
    bench = " (J&P benchmark)" if idx == 0 else ""
    return (
        f"Amostra {idx}{bench} | Hs={Hs:.2f} m, fp={fp:.3f} Hz, "
        f"gamma={gamma:.2f}, theta0={np.degrees(th0):.1f} deg, s={s}"
    )


def _polar_plot(ax, theta, f, data, cmap, vmin=None, vmax=None,
                title="", clabel="", symlog=False, linthresh=None):
    TH, R = np.meshgrid(theta, f)

    if vmin is None:
        vmin = float(data.min())
    if vmax is None:
        vmax = float(data.max())

    norm = None
    if symlog:
        lt = linthresh if linthresh is not None else max(abs(vmax), abs(vmin)) * 0.01
        lt = lt if lt > 0 else 1e-10
        norm = mcolors.SymLogNorm(linthresh=lt, vmin=vmin, vmax=vmax, base=10)

    pcm = ax.pcolormesh(
        TH, R, data, cmap=cmap, norm=norm,
        vmin=(None if norm else vmin),
        vmax=(None if norm else vmax),
        shading="auto",
    )
    plt.colorbar(pcm, ax=ax, label=clabel, pad=0.08)
    ax.set_title(title, pad=15, fontsize=10)
    return pcm


def plot_comparison(idx, E_2d, St_gt, St_pred, f, theta, fp, Hs, gamma, th0, s):
    """
    Compara S_tilde ground truth e predição na escala normalizada [-1, 1].
    """
    error    = St_pred - St_gt
    St_1d_gt   = _trapz(St_gt,   theta, axis=1)
    St_1d_pred = _trapz(St_pred, theta, axis=1)
    E_1d       = _trapz(E_2d,    theta, axis=1)

    fig = plt.figure(figsize=(22, 11))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle(
        _build_title(idx, fp, Hs, gamma, th0, s)
        + "\n[Comparação: FNO vs Ground Truth — escala normalizada]",
        fontsize=12, fontweight="bold",
    )

    vmax_st  = 1.0   # normalizado, sempre em [-1, 1]
    vmax_err = float(np.max(np.abs(error))) or 1e-30

    # (0,0) E(f, θ) — Input
    ax00 = fig.add_subplot(gs[0, 0], projection="polar")
    _polar_plot(ax00, theta, f, E_2d, "hot_r",
                title="E(f, θ) — Input", clabel="m² Hz⁻¹ rad⁻¹")

    # (0,1) S̃_nl — Ground Truth normalizado
    ax01 = fig.add_subplot(gs[0, 1], projection="polar")
    _polar_plot(ax01, theta, f, St_gt, "RdBu_r",
                vmin=-vmax_st, vmax=vmax_st,
                title="S̃_nl — Ground Truth (normalizado)",
                clabel="adimensional", symlog=True)

    # (0,2) S̃_nl — Predição FNO
    ax02 = fig.add_subplot(gs[0, 2], projection="polar")
    _polar_plot(ax02, theta, f, St_pred, "RdBu_r",
                vmin=-vmax_st, vmax=vmax_st,
                title="S̃_nl — Predição FNO (normalizado)",
                clabel="adimensional", symlog=True)

    # (0,3) Erro (FNO − GT)
    ax03 = fig.add_subplot(gs[0, 3], projection="polar")
    _polar_plot(ax03, theta, f, error, "PiYG",
                vmin=-vmax_err, vmax=vmax_err,
                title="Erro  (FNO − GT)",
                clabel="adimensional", symlog=True)

    # (1,0) E(f) integrado
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(f, E_1d, "k-", lw=2)
    ax10.axvline(fp, color="red", ls="--", lw=1, label=f"fp={fp:.2f} Hz")
    ax10.set_title("Espectro Integrado 1D  E(f)", fontsize=10)
    ax10.set_xlabel("f (Hz)")
    ax10.set_ylabel("E(f) (m² Hz⁻¹)")
    ax10.grid(True, ls=":", alpha=0.6)
    ax10.legend()

    # (1,1:4) Integral direcional S̃_nl(f)
    ax11 = fig.add_subplot(gs[1, 1:])
    ax11.plot(f, St_1d_gt,   "k-",  lw=2.0, label="Ground Truth (DE3)")
    ax11.plot(f, St_1d_pred, "r--", lw=1.8, label="FNO predito")
    ax11.axhline(0,  color="k",    ls="-",  lw=0.7)
    ax11.axvline(fp, color="blue", ls="--", lw=1.0, label=f"fp={fp:.2f} Hz")
    ax11.set_title("Integral Direcional  S̃_nl(f)  (normalizado)", fontsize=10)
    ax11.set_xlabel("f (Hz)")
    ax11.set_ylabel("S̃_nl(f)  (adimensional)")
    ax11.legend(fontsize=9)
    ax11.grid(True, ls=":", alpha=0.6)

    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("h5file", nargs="?", default="Generate_data/snl/snl_dataset.h5")
    pa.add_argument("--idx",            type=int,   default=0,
                    help="Índice da amostra")
    pa.add_argument("--model",          default="shape_prediction/results/model_best.pth",
                    help="Caminho do modelo FNO")
    pa.add_argument("--out-dir",        default="shape_prediction/figures")
    pa.add_argument("--n-modes",        type=int, nargs=2, default=[16, 16])
    pa.add_argument("--hidden-channels",type=int, default=64)
    args = pa.parse_args()

    # -- Carregar dados ------------------------------------------------------
    print("\n Carregando dados...")
    with h5py.File(args.h5file, "r") as hf:
        n_total = hf["X"].shape[0]
        idx = args.idx % n_total
        E    = hf["X"][idx, ..., 0]
        S_gt = hf["Y"][idx, ..., 0]
        f     = hf["f"][:]
        theta = hf["theta"][:]
        fp    = float(hf["fp"][idx])
        Hs    = float(hf["Hs"][idx])
        gamma = float(hf["gamma"][idx])
        th0   = float(hf["theta0"][idx])
        s     = int(hf["s"][idx])
    print(f" Amostra {idx} carregada")

    # -- Normalizar ground truth ---------------------------------------------
    a_gt = float(np.max(np.abs(S_gt)))
    St_gt = S_gt / a_gt
    print(f" Ground truth normalizado (a_gt = {a_gt:.6e})")

    # -- Carregar e rodar FNO ------------------------------------------------
    print(f"\n Carregando FNO: {args.model}")
    model = load_model(args.model, tuple(args.n_modes), args.hidden_channels)

    x = torch.tensor(E[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        St_pred_t = model(x)
    St_pred = St_pred_t[0, 0].cpu().numpy()
    print(f" Predição shape: {St_pred.shape}")

    # -- Métricas ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("MÉTRICAS — escala normalizada")
    print("=" * 60)
    print(f"Relative L2 (forma)  = {rel_l2(St_pred, St_gt):.6e}")
    print(f"MAE                  = {float(np.mean(np.abs(St_pred - St_gt))):.6e}")
    print(f"Max |erro|           = {float(np.max(np.abs(St_pred - St_gt))):.6e}")
    print("=" * 60)

    # -- Gráfico -------------------------------------------------------------
    print(f"\n Gerando gráfico...")
    os.makedirs(args.out_dir, exist_ok=True)

    fig = plot_comparison(idx, E, St_gt, St_pred, f, theta, fp, Hs, gamma, th0, s)

    out = os.path.join(args.out_dir, f"fno_shape_only_sample_{idx}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" Salvo: {out}")


if __name__ == "__main__":
    main()
