#!/usr/bin/env python3
"""
Inspeção do modelo FNO com amplitude externa:
  E(f, theta) -> S_tilde(f, theta) [FNO]
  E(f, theta) -> a [modelo LinearRegressor]
  S_pred = a_pred * S_tilde_pred
"""

import argparse
import json
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
EPS = 1e-8


class LinearRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)
    def forward(self, x):
        return self.linear(x)


class FNOOnly(nn.Module):
    def __init__(self, n_modes=(16, 16), hidden_channels=64):
        super().__init__()
        self.field_model = make_fno_2d(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=1,
            out_channels=1
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
        # Se NÃO tiver "field_model." no início, adiciona
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


def rel_l2(pred, target, eps=EPS):
    return float(np.linalg.norm((pred - target).ravel()) / (np.linalg.norm(target.ravel()) + eps))


def polar_plot(ax, theta, f, data, title, cmap="RdBu_r", symmetric=True, symlog=True):
    TH, R = np.meshgrid(theta, f)
    if symmetric:
        vmax = float(np.max(np.abs(data))) or 1e-30
        vmin = -vmax
    else:
        vmin = float(np.min(data)); vmax = float(np.max(data))
    norm = None
    if symlog:
        norm = mcolors.SymLogNorm(linthresh=max(abs(vmin), abs(vmax)) * 0.01 + 1e-30, vmin=vmin, vmax=vmax, base=10)
    pcm = ax.pcolormesh(TH, R, data, cmap=cmap, norm=norm, vmin=None if norm else vmin, vmax=None if norm else vmax, shading="auto")
    ax.set_title(title, pad=14, fontsize=10)
    plt.colorbar(pcm, ax=ax, pad=0.08)


def build_title(idx, fp, Hs, gamma, th0, s):
    bench = " (J&P benchmark)" if idx == 0 else ""
    return f"Amostra {idx}{bench} | Hs={Hs:.2f} m, fp={fp:.3f} Hz, gamma={gamma:.2f}, theta0={np.degrees(th0):.1f} deg, s={s}"


def plot_comparison(idx, E, S_gt, S_pred, St_gt, St_pred, f, theta, fp, Hs, gamma, th0, s, a_gt, a_pred):
    err = S_pred - S_gt
    S_gt_1d = _trapz(S_gt, theta, axis=1)
    S_pred_1d = _trapz(S_pred, theta, axis=1)
    rel_phys = rel_l2(S_pred, S_gt)
    rel_shape = rel_l2(St_pred, St_gt)
    rel_a = abs(a_pred - a_gt) / (abs(a_gt) + EPS)

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle(build_title(idx, fp, Hs, gamma, th0, s) +
                 f"\na_gt={a_gt:.3e}, a_pred={a_pred:.3e}, err_a={rel_a:.3e}, shape RelL2={rel_shape:.3e}, physical RelL2={rel_phys:.3e}",
                 fontsize=12, fontweight="bold")

    # Linha 0: Campos físicos
    ax = fig.add_subplot(gs[0, 0], projection="polar")
    polar_plot(ax, theta, f, E, "E(f, theta) input", cmap="hot_r", symmetric=False, symlog=False)
    
    ax = fig.add_subplot(gs[0, 1], projection="polar")
    polar_plot(ax, theta, f, S_gt, "S_nl ground truth")
    
    ax = fig.add_subplot(gs[0, 2], projection="polar")
    polar_plot(ax, theta, f, S_pred, "S_nl reconstruído (FNO * a)")
    
    ax = fig.add_subplot(gs[0, 3], projection="polar")
    polar_plot(ax, theta, f, err, "Erro físico (pred - GT)", cmap="PiYG")

    # Linha 1: Campos normalizados
    ax = fig.add_subplot(gs[1, 0], projection="polar")
    polar_plot(ax, theta, f, St_gt, "S_nl normalizado GT")
    
    ax = fig.add_subplot(gs[1, 1], projection="polar")
    polar_plot(ax, theta, f, St_pred, "S_nl normalizado pred (FNO)")

    # Integral direcional
    ax = fig.add_subplot(gs[1, 2:])
    ax2 = ax.twinx()
    ax2.plot(f, S_gt_1d, "k-", lw=2, label="S_nl GT")
    ax2.plot(f, S_pred_1d, "r--", lw=1.8, label="S_nl pred (FNO * a)")
    ax.axvline(fp, color="blue", ls="--", lw=1, label=f"fp={fp:.3f} Hz")
    ax.set_xlabel("f (Hz)")
    ax.set_ylabel("E(f)")
    ax2.set_ylabel("Integral direcional de S_nl")
    ax.grid(True, ls=":", alpha=0.5)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=9)
    ax.set_title("Integral direcional")
    
    return fig, {"rel_phys": rel_phys, "rel_shape": rel_shape, "rel_a": rel_a}


def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("h5file", nargs="?", default="Generate_data/snl_data/snl_dataset.h5")
    pa.add_argument("--idx", type=int, default=0, help="Índice da amostra")
    pa.add_argument("--model", default="fno/results_snl/model_best.pth", help="Caminho do modelo FNO")
    pa.add_argument("--amplitude-model", default="scale/results_scale_loggamma_s/model_best.pth", help="Caminho do modelo de amplitude")
    pa.add_argument("--out-dir", default="fno_and_a/results_snl_with_amplitude")
    pa.add_argument("--n-modes", type=int, nargs=2, default=[16, 16])
    pa.add_argument("--hidden-channels", type=int, default=64)
    args = pa.parse_args()

    print("\n Carregando dados...")
    with h5py.File(args.h5file, "r") as hf:
        n_total = hf["X"].shape[0]
        idx = args.idx % n_total
        E = hf["X"][idx, ..., 0]
        S_gt = hf["Y"][idx, ..., 0]
        f = hf["f"][:]
        theta = hf["theta"][:]
        fp = float(hf["fp"][idx])
        Hs = float(hf["Hs"][idx])
        gamma = float(hf["gamma"][idx])
        th0 = float(hf["theta0"][idx])
        s = int(hf["s"][idx])
    print(f" Amostra {idx} carregada")


    print(f"\n Carregando modelo de amplitude: {args.amplitude_model}")
    amplitude_model = LinearRegressor()
    amplitude_model.load_state_dict(torch.load(args.amplitude_model, map_location="cpu"))
    amplitude_model.eval()

    Hs_tensor = torch.tensor([Hs], dtype=torch.float32)
    fp_tensor = torch.tensor([fp], dtype=torch.float32)
    gamma_tensor = torch.tensor([gamma], dtype=torch.float32)
    s_tensor = torch.tensor([s], dtype=torch.float32)

    X_amp = torch.stack([
        torch.log10(Hs_tensor + EPS),
        torch.log10(fp_tensor + EPS),
        torch.log10(gamma_tensor + EPS),
        s_tensor
    ], dim=1)


    with torch.no_grad():
        a_pred_log = amplitude_model(X_amp)
        a_pred = 10 ** a_pred_log.numpy()[0, 0]
    print(f" Amplitude prevista: a_pred = {a_pred:.6e}")


    print(f"\n Carregando FNO: {args.model}")
    model = load_model(args.model, tuple(args.n_modes), args.hidden_channels)


    x = torch.tensor(E[np.newaxis, np.newaxis, :, :], dtype=torch.float32)
    with torch.no_grad():
        St_pred_t = model(x)
    St_pred = St_pred_t[0, 0].cpu().numpy()
    print(f" Forma normalizada prevista: St_pred shape {St_pred.shape}")

    # ==
    a_gt = float(np.max(np.abs(S_gt)) + EPS)
    St_gt = S_gt / a_gt
    S_pred = St_pred * a_pred


    print("\n" + "="*60)
    print("MÉTRICAS")
    print("="*60)
    print(f"a_gt     = {a_gt:.6e}")
    print(f"a_pred   = {a_pred:.6e}")
    print(f"rel_a    = {abs(a_pred-a_gt)/(abs(a_gt)+EPS):.6e} ({abs(a_pred-a_gt)/(abs(a_gt)+EPS)*100:.2f}%)")
    print(f"shape RelL2   = {rel_l2(St_pred, St_gt):.6e}")
    print(f"physical RelL2 = {rel_l2(S_pred, S_gt):.6e}")
    print("="*60)

    print(f"\nGerando gráfico...")
    os.makedirs(args.out_dir, exist_ok=True)
    
    fig, _ = plot_comparison(idx, E, S_gt, S_pred, St_gt, St_pred, f, theta, fp, Hs, gamma, th0, s, a_gt, a_pred)
    
    out = os.path.join(args.out_dir, f"fno_with_amplitude_sample_{idx}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" Salvo: {out}")

if __name__ == "__main__":
    main()