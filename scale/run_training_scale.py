#!/usr/bin/env python3
"""
Treinamento apenas da amplitude (escala a) a partir de E(f, theta).

  - A fatorização separa:
        a = max_{f,theta} |S_nl|
  - Uma FFN (ScaleHead) aprende a amplitude:
        E -> a
  - A amplitude é treinada em escala logarítmica
"""

import argparse
import json
import os
import yaml

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


DATA_PATH = "Generate_data/snl_data/snl_dataset.h5"
RUN_DIR = "scale/results_scale"
EPS = 1e-8


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def scale_log_mse(pred_a, true_a, eps=EPS):
    """MSE no log da amplitude, para estabilizar variações de ordem de grandeza."""
    return torch.mean((torch.log(pred_a + eps) - torch.log(true_a + eps)) ** 2)

def scale_mse(pred_a, true_a):
    """MSE simples para log(a)."""
    return torch.mean((pred_a - true_a) ** 2)


# ---------------------------------------------------------------------------
# Fatorização
# ---------------------------------------------------------------------------

def factorize_target(Y, eps=EPS):
    """
    Y: (N, 1, Nf, Ntheta)
    Retorna:
      a: max_abs(Y), shape (N, 1)
    """
    a = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + eps
    return a.view(-1, 1)


# ---------------------------------------------------------------------------
# Modelo: ScaleHead
# ---------------------------------------------------------------------------

class ScaleHead(nn.Module):
    """MLP que prevê amplitude a partir de (Hs, fp)"""
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden),          # entrada: Hs, fp
            nn.GELU(),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, 1),
            nn.Softplus(),                 # amplitude positiva
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------
'''
 hf.create_dataset("X", data=X, compression="gzip")
        hf.create_dataset("Y", data=Y, compression="gzip")
hf.create_dataset("f", data=f); hf.create_dataset("theta", data=theta)
        hf.create_dataset("fp", data=fp_a); hf.create_dataset("Hs", data=hs_a)
        hf.create_dataset("gamma", data=gamma_a); hf.create_dataset("theta0", data=th0_a)
        hf.create_dataset("s", data=s_a)
'''

def load_snl_dataset(path):
    with h5py.File(path, "r") as hf:
        Hs = hf["Hs"][:]
        Fp =hf["fp"][:]
        Y = hf["Y"][:]
    Hs = torch.tensor(Hs, dtype=torch.float32)
    Fp = torch.tensor(Fp, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)
 
    return Hs,Fp, Y

def load_hs_fp_dataset(path):
   
    with h5py.File(path, "r") as hf:
        Hs = hf["Hs"][:]   # (N,)
        fp = hf["fp"][:]   # (N,)
        gamma = hf["gamma"][:]   # (N,)
        s = hf["s"][:]   # (N,)
        Y = hf["Y"][:]
    
    Hs = torch.tensor(Hs, dtype=torch.float32)
    fp = torch.tensor(fp, dtype=torch.float32)
    s = torch.tensor(s, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)
    
    return Hs, fp,Y, gamma,s


def compute_stats(tensor):
    return {
        "min": float(tensor.min()),
        "max": float(tensor.max()),
        "mean": float(tensor.mean()),
        "std": float(tensor.std()),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def save_curve(train_values, val_values, out_path, ylabel, title):
    plt.figure(figsize=(7, 4.5))
    plt.plot(train_values, label="Train")
    plt.plot(val_values, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

class LinearRegressor(nn.Module):
    #y = w1*x1 + w2*x2 + b
    #sem camadas ou função de ativação
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)  # 2 entradas (Hs, fp), 1 saída (log(a))

    def forward(self, x):
        return self.linear(x)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--h5file", default=DATA_PATH)
    pa.add_argument("--out-dir", default=RUN_DIR)
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch", type=int, default=16)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--scale-head-hidden", type=int, default=128)
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)

    config = {
        "model": {
            #LinearRegressor
            "type": "LinearRegressor",
            "scale_head_hidden": args.scale_head_hidden,
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch,
            "learning_rate": args.lr,
            "scale_loss": "log_MSE",
        },
        "data": {"path": args.h5file},
    }
    with open(os.path.join(args.out_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f, sort_keys=False)

    hs, fp, Y,gamma,s = load_hs_fp_dataset(args.h5file)


    a = factorize_target(Y)
    a = torch.log10(a + EPS)  
    
    

    X = torch.stack([torch.log10(hs+EPS), torch.log10(fp+EPS),gamma,s], dim=1)
   # X = torch.log(X + EPS)  



    n_total = len(X)
    n_train = int(0.8 * n_total)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    train_ds = TensorDataset(X[train_idx], a[train_idx])
    val_ds = TensorDataset(X[val_idx], a[val_idx])
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch)
#LinearRegressor
    model = LinearRegressor().to(device)

    optimizer = Adam(model.parameters(), lr=args.lr)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_losses, val_losses = [], []
    best_val = float("inf")

    for epoch in range(args.epochs):
        model.train()
        tr_loss = 0.0

        for xb, ab in train_loader:
            xb = xb.to(device)
            ab = ab.to(device)

            optimizer.zero_grad()
            pred_a = model(xb)

            loss =scale_mse(pred_a, ab)
            loss.backward()
            optimizer.step()

            tr_loss += loss.item() * xb.size(0)

        tr_loss /= len(train_loader.dataset)

        model.eval()
        vl_loss = 0.0
        with torch.no_grad():
            for xb, ab in val_loader:
                xb = xb.to(device)
                ab = ab.to(device)

                pred_a = model(xb)
                loss =scale_mse(pred_a, ab)
                vl_loss += loss.item() * xb.size(0)

        vl_loss /= len(val_loader.dataset)

        train_losses.append(tr_loss)
        val_losses.append(vl_loss)

        scheduler.step(vl_loss)

        if vl_loss < best_val:
            best_val = vl_loss
            torch.save(model.state_dict(), os.path.join(args.out_dir, "model_best.pth"))

        print(f"Epoch {epoch:03d} | Train loss: {tr_loss:.4e} | Val loss: {vl_loss:.4e}")

    torch.save(model.state_dict(), os.path.join(args.out_dir, "model.pth"))

    # Depois de treinar o modelo

# Pegar os pesos e bias
    w = model.linear.weight.detach().cpu().numpy().flatten()
    b = model.linear.bias.detach().cpu().numpy().item()

    print(f"Coeficientes:")
    print(f"  w1 = {w[0]:.6f}  (log(Hs))")
    print(f"  w2 = {w[1]:.6f}  (log(fp))")
    print(f"  b  = {b:.6f}")

    metrics = {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val,
        "train_stats": {
            "a": compute_stats(a[train_idx]),
        },
        "config": config,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    save_curve(train_losses, val_losses, os.path.join(args.out_dir, "loss_scale.pdf"), "Log-MSE", "Scale loss: log-MSE em a")

    model.eval()
    all_true = []
    all_pred = []

    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            pred = model(xb)
            all_true.extend(yb.cpu().numpy().flatten())
            all_pred.extend(pred.cpu().numpy().flatten())
            
     #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    # Scatter plot
    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())

    plt.figure(figsize=(6, 5))
    plt.scatter(all_true, all_pred, alpha=0.3, s=5)

    # Eixos com mesma escala
    plt.xlim(min_val - 0.5, max_val + 0.5)
    plt.ylim(min_val - 0.5, max_val + 0.5)
    plt.axis('equal')

    # Linha ideal y = x
    plt.plot([min_val - 0.5, max_val + 0.5],
             [min_val - 0.5, max_val + 0.5],
             'r--', lw=2, label='Ideal')

    plt.xlabel('log(Amplitude Real)')
    plt.ylabel('log(Amplitude Predita)')
    plt.title(f'Real vs Predito - {len(all_true)} amostras')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "amplitude_scatter_gammaessemlog.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter salvo em: {os.path.join(args.out_dir, 'amplitude_scatter_gammaessemlog.pdf')}")

    print(f"\n Treinamento finalizado!")
    print(f" Arquivos salvos em: {args.out_dir}")
    print(f" Best validation loss: {best_val:.4e}")
    
    #AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
        # Grafico com exponenciacao base 10
    # Grafico com exponenciacao base 10 - zoom em 0-2
    all_true_exp = 10 ** all_true
    all_pred_exp = 10 ** all_pred

    plt.figure(figsize=(6, 5))
    plt.scatter(all_true_exp, all_pred_exp, alpha=0.3, s=5)

    # Zoom fixo de 0 a 2
    plt.xlim(0, 2)
    plt.ylim(0, 2)
    plt.axis('equal')

    # Linha ideal no intervalo do zoom
    plt.plot([0, 2], [0, 2], 'r--', lw=2, label='Ideal')

    plt.xlabel('Amplitude Real')
    plt.ylabel('Amplitude Predita')
    plt.title(f'Real vs Predito (zoom 0-2) - {len(all_true)} amostras')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "amplitude_scatter_zoom_0_2.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    
    

if __name__ == "__main__":
    main()