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
import joblib 
from sklearn.linear_model import LinearRegression 

from fno_diffusion.model import make_fno_2d

_trapz = getattr(np, "trapezoid", None) or np.trapz
EPS = 1e-8


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


def load_amplitude_model(model_path):
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "amplitude_model_params.pth")
        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f" amplitude_model_params.pth não encontrado em {model_path}"
            )

    if not model_path.endswith('.pth'):
        raise ValueError(f"Arquivo deve ser .pth: {model_path}")
    
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
    
    model = LinearRegression()
    model.coef_ = checkpoint['coef_'].numpy()
    model.intercept_ = checkpoint['intercept_'].numpy()
    
    print(f" Modelo carregado de: {model_path}")
    print(f"   Coeficientes: {model.coef_}")
    print(f"   Intercept: {model.intercept_:.6f}")
    print(f"   Features: {checkpoint.get('feature_names', 'N/A')}")
    
    return model

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


def _build_title(idx, fp, Hs, gamma, th0, s):
    bench = " (J&P benchmark)" if idx == 0 else ""
    return f"Amostra {idx}{bench} | Hs={Hs:.2f} m, fp={fp:.3f} Hz, gamma={gamma:.2f}, theta0={np.degrees(th0):.1f} deg, s={s}"

def _polar_plot(ax, theta, f, data, cmap, vmin=None, vmax=None, title="", clabel="",
               symlog=False, linthresh=None):
    """
    Plot polar com suporte a SymLogNorm para dados esparsos (quase tudo zero,
    com picos localizados) - caso típico do S_nl.

    Se symlog=True (ativado automaticamente para colormaps divergentes quando
    os dados têm amplitude muito pequena), usa escala logarítmica simétrica
    para tornar os picos visíveis mesmo quando a maioria dos valores é ~0.
    """
    TH, R = np.meshgrid(theta, f)

    if vmin is None: vmin = float(data.min())
    if vmax is None: vmax = float(data.max())

    norm = None
    if symlog:
        # linthresh: faixa linear ao redor do zero (evita log(0))
        # Se não fornecido, usa 1% do valor máximo absoluto
        lt = linthresh if linthresh is not None else max(abs(vmax), abs(vmin)) * 0.01
        lt = lt if lt > 0 else 1e-10
        norm = mcolors.SymLogNorm(linthresh=lt, vmin=vmin, vmax=vmax, base=10)

    pcm = ax.pcolormesh(TH, R, data, cmap=cmap, norm=norm,
                        vmin=(None if norm else vmin),
                        vmax=(None if norm else vmax),
                        shading="auto")
    plt.colorbar(pcm, ax=ax, label=clabel, pad=0.08)
    ax.set_title(title, pad=15, fontsize=10)
    return pcm

def plot_comparison(idx, E_2d, Snl_gt, Snl_pred, f, theta, fp, Hs, gamma, th0, s):
    error        = Snl_pred - Snl_gt          # erro na escala física
    Snl_1d_gt   = _trapz(Snl_gt,   theta, axis=1)
    Snl_1d_pred = _trapz(Snl_pred, theta, axis=1)
    E_1d        = _trapz(E_2d,     theta, axis=1)

    # Layout: 2 linhas x 4 colunas
    fig = plt.figure(figsize=(22, 11))
    gs  = GridSpec(2, 4, figure=fig, hspace=0.45, wspace=0.4)
    fig.suptitle(
        _build_title(idx, fp, Hs, gamma, th0, s) + "\n[Comparação: FNO vs Ground Truth]",
        fontsize=12, fontweight="bold"
    )

    vmax_snl = float(max(np.max(np.abs(Snl_gt)), np.max(np.abs(Snl_pred)))) or 1e-30
    vmax_err = float(np.max(np.abs(error))) or 1e-30

    # (0,0) Polar E - Input
    ax00 = fig.add_subplot(gs[0, 0], projection="polar")
    _polar_plot(ax00, theta, f, E_2d, "hot_r",
                title="E(f, θ) — Input", clabel="m² Hz⁻¹ rad⁻¹")

    # (0,1) Polar Snl GT
    ax01 = fig.add_subplot(gs[0, 1], projection="polar")
    _polar_plot(ax01, theta, f, Snl_gt, "RdBu_r",
                vmin=-vmax_snl, vmax=vmax_snl,
                title="S_nl — Ground Truth (DE3)",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹", symlog=True)

    # (0,2) Polar Snl FNO
    ax02 = fig.add_subplot(gs[0, 2], projection="polar")
    _polar_plot(ax02, theta, f, Snl_pred, "RdBu_r",
                vmin=-vmax_snl, vmax=vmax_snl,
                title="S_nl — Predição FNO",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹", symlog=True)

    # (0,3) Polar Erro (FNO − GT) - colormap divergente centrado em zero
    ax03 = fig.add_subplot(gs[0, 3], projection="polar")
    _polar_plot(ax03, theta, f, error, "PiYG",
                vmin=-vmax_err, vmax=vmax_err,
                title="Erro  (FNO − GT)",
                clabel="m² Hz⁻¹ rad⁻¹ s⁻¹", symlog=True)

    # (1,0) 1D E(f)
    ax10 = fig.add_subplot(gs[1, 0])
    ax10.plot(f, E_1d, "k-", lw=2)
    ax10.axvline(fp, color="red", ls="--", lw=1, label=f"fp={fp:.2f} Hz")
    ax10.set_title("Espectro Integrado 1D  E(f)", fontsize=10)
    ax10.set_xlabel("f (Hz)"); ax10.set_ylabel("E(f) (m² Hz⁻¹)")
    ax10.grid(True, ls=":", alpha=0.6); ax10.legend()

    # (1,1:4) 1D Snl comparativo - ocupa as 3 colunas restantes
    ax11 = fig.add_subplot(gs[1, 1:])
    ax11.plot(f, Snl_1d_gt,   "k-",  lw=2.0, label="Ground Truth (DE3)")
    ax11.plot(f, Snl_1d_pred, "r--", lw=1.8, label="FNO predito")
    ax11.axhline(0,  color="k",    ls="-",  lw=0.7)
    ax11.axvline(fp, color="blue", ls="--", lw=1.0, label=f"fp={fp:.2f} Hz")
    ax11.set_title("Integral Direcional  S_nl(f)", fontsize=10)
    ax11.set_xlabel("f (Hz)"); ax11.set_ylabel("S_nl(f) (m² Hz⁻¹ s⁻¹)")
    ax11.legend(fontsize=9); ax11.grid(True, ls=":", alpha=0.6)

    return fig

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("h5file", nargs="?", default="Generate_data/snl/snl_dataset.h5")
    pa.add_argument("--idx", type=int, default=0, help="Índice da amostra")
    pa.add_argument("--model", default="shape_prediction/results/model_best.pth", help="Caminho do modelo FNO")

    #incompleto
    pa.add_argument("--amplitude-model", default="amplitude_prediction/linear_regression/results/amplitude_model_params.pth", help="Caminho do modelo de amplitude")
    pa.add_argument("--out-dir", default="shape_prediction/figures")
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
    amplitude_model = load_amplitude_model(args.amplitude_model)

    Hs_tensor = torch.tensor([Hs], dtype=torch.float32)
    fp_tensor = torch.tensor([fp], dtype=torch.float32)
    gamma_tensor = torch.tensor([gamma], dtype=torch.float32)
    s_tensor = torch.tensor([s], dtype=torch.float32)

    X_amp = np.array([[
        np.log10(Hs + EPS),
        np.log10(fp + EPS),
        np.log10(gamma + EPS),
        float(s)
    ]])


    a_pred_log = amplitude_model.predict(X_amp)[0]
    a_pred = 10 ** a_pred_log
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

    #def plot_comparison(idx, E_2d, Snl_gt, Snl_pred, f, theta, fp, Hs, gamma, th0, s):
    
    fig = plot_comparison(idx, E, S_gt, S_pred, f, theta, fp, Hs, gamma, th0, s)
    
    out = os.path.join(args.out_dir, f"fno_with_amplitude_sample_{idx}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f" Salvo: {out}")

if __name__ == "__main__":
    main()