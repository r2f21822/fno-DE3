import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.linear_model import LinearRegression
import os

EPS = 1e-8

DATA_PATH = "Generate_data/snl/snl_dataset.h5"
RUN_DIR   = "Generate_data"

def factorize_target(Y, eps=EPS):
    a = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + eps
    return a.view(-1, 1)

def load_snl_dataset(path):
    with h5py.File(path, "r") as hf:
        s = hf["s"][:]
        gamma = hf["gamma"][:]
        Y = hf["Y"][:]
    gamma = torch.tensor(gamma, dtype=torch.float32)
    s = torch.tensor(s, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)
    return gamma, s, Y

def main():
    gamma, s, Y = load_snl_dataset(DATA_PATH)

    a = factorize_target(Y)
    loga = torch.log(a + EPS)  

    loga_np = loga.numpy().flatten()
    gamma_np = gamma.numpy().flatten()
    s_np = s.numpy().flatten()

    reg_gamma = LinearRegression().fit(gamma_np.reshape(-1, 1), loga_np)
    slope_gamma = reg_gamma.coef_[0]
    intercept_gamma = reg_gamma.intercept_
    r2_gamma = reg_gamma.score(gamma_np.reshape(-1, 1), loga_np)

    print("REGRESSÃO: log(a) vs gamma")
    print(f"  Inclinacao: {slope_gamma:.4f}")
    print(f"  Intercepto: {intercept_gamma:.4f}")
    print(f"  R2: {r2_gamma:.4f}")

    reg_s = LinearRegression().fit(s_np.reshape(-1, 1), loga_np)
    slope_s = reg_s.coef_[0]
    intercept_s = reg_s.intercept_
    r2_s = reg_s.score(s_np.reshape(-1, 1), loga_np)

    print("\nREGRESSÃO: log(a) vs s")
    print(f"  Inclinacao: {slope_s:.4f}")
    print(f"  Intercepto: {intercept_s:.4f}")
    print(f"  R2: {r2_s:.4f}")

    plt.figure(figsize=(6, 5))
    plt.scatter(gamma_np, loga_np, alpha=0.3, s=5)
    plt.xlabel('gamma')
    plt.ylabel('log(a)')
    plt.title(f'log(a) vs gamma - {len(loga)} amostras')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "log_a_vs_gamma.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo em: {os.path.join(RUN_DIR, 'log_a_vs_gamma.pdf')}")

    plt.figure(figsize=(6, 5))
    plt.scatter(s_np, loga_np, alpha=0.3, s=5)
    plt.xlabel('s')
    plt.ylabel('log(a)')
    plt.title(f'log(a) vs s - {len(loga)} amostras')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "log_a_vs_s.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo em: {os.path.join(RUN_DIR, 'log_a_vs_s.pdf')}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(gamma_np, loga_np, alpha=0.3, s=5)
    axes[0].set_xlabel('gamma')
    axes[0].set_ylabel('log(a)')
    axes[0].set_title(f'log(a) vs gamma (R2={r2_gamma:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].scatter(s_np, loga_np, alpha=0.3, s=5)
    axes[1].set_xlabel('s')
    axes[1].set_ylabel('log(a)')
    axes[1].set_title(f'log(a) vs s (R2={r2_s:.4f})')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "log_a_vs_gamma_and_s.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Salvo em: {os.path.join(RUN_DIR, 'log_a_vs_gamma_and_s.pdf')}")

    print(f"\nAnalise finalizada")
    print(f"Arquivos salvos em: {RUN_DIR}")

if __name__ == "__main__":
    main()