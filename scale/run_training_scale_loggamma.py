
#log de gamma e s normal
#arquivo principal

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
RUN_DIR = "scale/results_scale_loggamma_s"
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
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1) 

    def forward(self, x):
        return self.linear(x)



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


    hs, fp, Y, gamma, s = load_hs_fp_dataset(args.h5file)
    
    a = factorize_target(Y)
    a = torch.log10(a + EPS)
    
    X = torch.stack([torch.log10(hs+EPS), torch.log10(fp+EPS), 
                     torch.log10(gamma), s], dim=1)
    
    # Dividir em treino/validação
    n_total = len(X)
    n_train = int(0.8 * n_total)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(args.seed))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]
    
    X_train = X[train_idx]
    y_train = a[train_idx]
    X_val = X[val_idx]
    y_val = a[val_idx]
    
    X_train_com_bias = torch.cat([torch.ones(X_train.shape[0], 1), X_train], dim=1)

    beta = torch.linalg.lstsq(X_train_com_bias, y_train).solution
    
 
    b = beta[0].item()  
    w = beta[1:].flatten().cpu().numpy()  
    
    print(f"Coeficientes encontrados:")
    print(f"  w1 = {w[0]:.6f}  (log(Hs))")
    print(f"  w2 = {w[1]:.6f}  (log(fp))")
    print(f"  w3 = {w[2]:.6f}  (log(gamma))")
    print(f"  w4 = {w[3]:.6f}  (s)")
    print(f"  b  = {b:.6f}")
    
  
    X_val_com_bias = torch.cat([torch.ones(X_val.shape[0], 1), X_val], dim=1)
    y_pred = X_val_com_bias @ beta
    

    mse = torch.mean((y_pred - y_val) ** 2).item()
    print(f"MSE na validação: {mse:.6f}")

    print("MSE:")


    all_true = y_val.cpu().numpy().flatten()  
    all_pred = y_pred.cpu().numpy().flatten()
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)

    np.save(os.path.join(args.out_dir, "all_pred.npy"), all_pred)
    np.save(os.path.join(args.out_dir, "all_true.npy"), all_true)
    print(f"Previsões salvas em: {args.out_dir}")


    print(f"Previsões salvas em: {args.out_dir}")

    min_val = min(all_true.min(), all_pred.min())
    max_val = max(all_true.max(), all_pred.max())

    plt.figure(figsize=(6, 5))
    plt.scatter(all_true, all_pred, alpha=0.3, s=5)

    plt.xlim(min_val - 0.5, max_val + 0.5)
    plt.ylim(min_val - 0.5, max_val + 0.5)
    plt.axis('equal')


    plt.plot([min_val - 0.5, max_val + 0.5],
             [min_val - 0.5, max_val + 0.5],
             'r--', lw=2, label='Ideal')

    plt.xlabel('log(Amplitude Real)')
    plt.ylabel('log(Amplitude Predita)')
    plt.title(f'Real vs Predito - {len(all_true)} amostras')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, "amplitude_emLog_loggamma_s.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter salvo em: {os.path.join(args.out_dir, 'amplitude_emLog_loggamma_s.pdf')}")

    print(f"\n Treinamento finalizado!")
    print(f" Arquivos salvos em: {args.out_dir}")
 



    print("log(A) - Erro Medio Absoluto: ",np.mean(np.abs(all_true - all_pred)))  # erro médio absoluto
    print("log(A) - Erro Quadratico Médio :",np.mean((all_true - all_pred) ** 2))   
    print("log(A) - Raiz Erro Quadratico Medio",np.sqrt(np.mean((all_true - all_pred) ** 2)))                         # Raiz do erro Quadrático medio

        # Grafico com exponenciacao base 10
    all_true_exp1 = 10 ** all_true
    all_pred_exp1 = 10 ** all_pred

    print("A - Erro Medio Absoluto: ",np.mean(np.abs(all_true_exp1 - all_pred_exp1)))  
    print("A - Erro Quadratico Médio :",np.mean((all_true_exp1 - all_pred_exp1) ** 2))   
    print("A - Raiz Erro Quadratico Medio",np.sqrt(np.mean((all_true_exp1 - all_pred_exp1) ** 2))) 

    rmse_percentual = (np.sqrt(np.mean((all_true_exp1 - all_pred_exp1) ** 2)) / np.mean(all_true_exp1)) * 100
    print(f"RMSE percentual: {rmse_percentual}%")


    listaLimitaoes=[np.max(all_true),1,0.1,0.001,0.0001]
    
    for limitacao in listaLimitaoes:
        mask = (all_true_exp1 >= 0) & (all_true_exp1 <= limitacao) & (all_pred_exp1 >= 0) & (all_pred_exp1 <= limitacao)
        all_true_exp = all_true_exp1[mask]
        all_pred_exp = all_pred_exp1[mask]

        min_val_exp = min(all_true_exp.min(), all_pred_exp.min())
        max_val_exp = max(all_true_exp.max(), all_pred_exp.max())

        plt.figure(figsize=(6, 5))
        plt.scatter(all_true_exp, all_pred_exp, alpha=0.3, s=5)

        plt.xlim(min_val_exp * 0.9, max_val_exp * 1.1)
        plt.ylim(min_val_exp * 0.9, max_val_exp * 1.1)

        plt.axis('equal')

        plt.plot([min_val_exp * 0.9, max_val_exp * 1.1],
                [min_val_exp * 0.9, max_val_exp * 1.1],
                'r--', lw=2, label='Ideal')

        plt.xlabel('Amplitude Real (10^x)')
        plt.ylabel('Amplitude Predita (10^x)')
        plt.title(f'Real vs Predito (escala original) - {len(all_true)} amostras')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        nome_grafico_limitado = "amplitude_original_limitado_ate" + str(limitacao) + ".pdf"

        plt.savefig(os.path.join(args.out_dir, nome_grafico_limitado), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Scatter (escala original) salvo em: {os.path.join(args.out_dir, nome_grafico_limitado)}")


    listaLimitaoesPorAmostra=[1000,500,300,50]
    
    for min_amostras in listaLimitaoesPorAmostra:
        
        valores_ordenados = np.sort(all_true_exp1)
    
        limite_amplitude = valores_ordenados[min_amostras - 1]
        print(limite_amplitude)

        mask = (all_true_exp1 >= 0) & (all_true_exp1 <= limite_amplitude) & (all_pred_exp1 >= 0) & (all_pred_exp1 <= limite_amplitude)
        all_true_exp = all_true_exp1[mask]
        all_pred_exp = all_pred_exp1[mask]

        min_val_exp = min(all_true_exp.min(), all_pred_exp.min())
        max_val_exp = max(all_true_exp.max(), all_pred_exp.max())

        plt.figure(figsize=(6, 5))
        plt.scatter(all_true_exp, all_pred_exp, alpha=0.3, s=5)

        plt.xlim(min_val_exp * 0.9, max_val_exp * 1.1)
        plt.ylim(min_val_exp * 0.9, max_val_exp * 1.1)

        plt.axis('equal')

        plt.plot([min_val_exp * 0.9, max_val_exp * 1.1],
                [min_val_exp * 0.9, max_val_exp * 1.1],
                'r--', lw=2, label='Ideal')

        plt.xlabel('Amplitude Real (10^x)')
        plt.ylabel('Amplitude Predita (10^x)')
        plt.title(f'Real vs Predito (escala original) - treinado com {len(all_true)} amostras')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        nome_grafico_limitado = "amplitude_original_limitado_ate" + str(min_amostras) + "amostras.pdf"

        plt.savefig(os.path.join(args.out_dir, nome_grafico_limitado), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Scatter (escala original) salvo em: {os.path.join(args.out_dir, nome_grafico_limitado)}")
        

if __name__ == "__main__":
    main()