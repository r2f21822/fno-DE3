import argparse
import json
import os
import yaml
import h5py
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


DATA_PATH = "Generate_data/snl/snl_dataset_hs_variable.h5"
RUN_DIR = "snl_hs_dependence/results"
FIG_DIR = "snl_hs_dependence/figures"
IDX_DIR = "train_test_split/results_hs"
MODELO_DIR = "snl_hs_dependence/results"
RES_DIR="snl_hs_dependence/results"


def load_shared_idx(idx_dir=IDX_DIR):
    train_idx = np.load(os.path.join(idx_dir, "train_indices.npy"))
    val_idx = np.load(os.path.join(idx_dir, "val_indices.npy"))
    return train_idx, val_idx


def load_hs_fp_dataset(path):
    with h5py.File(path, "r") as hf:
        Hs = hf["Hs"][:]    
        Y = hf["Y"][:]
        fp = hf["fp"][:]
        gamma = hf["gamma"][:]
        theta0 = hf["theta0"][:]
        s = hf["s"][:]
        
        # Usa tolerância para comparar valores float
        mask = (np.isclose(fp, 0.1, atol=1e-6)) & \
               (np.isclose(gamma, 3.3, atol=1e-6)) & \
               (np.isclose(theta0, 0.0, atol=1e-6)) & \
               (s == 4)
        
        indices_validos = np.where(mask)[0]
        print(f"Amostras com parâmetros fixos: {len(indices_validos)}")
    
    Hs = torch.tensor(Hs, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)
    
    tam_Y = len(Y)
    
    return Hs, Y, tam_Y, indices_validos


def factorize_target(Y):
    """
    Y: (N, 1, Nf, Ntheta)
    Retorna:
      a: max_abs(Y), shape (N, 1)
    """
    a = Y.abs().amax(dim=(1, 2, 3), keepdim=True)
    for i, val in enumerate(a.view(-1)):
        if not torch.isfinite(val):
            raise ValueError(f"Amostra {i} invalida: amplitude não é finita (valor: {val})")
        if val <= 0:
            raise ValueError(f"Amostra {i} invalida: amplitude menor que zero (valor: {val})")
    
    shape = Y / a
    
    for i in range(len(shape)):
        max_abs = shape[i].abs().max().item()
        if not (max_abs == 1):
            raise ValueError(f"Amostra {i} inválida: max(abs(shape)) = {max_abs}, deveria ser 1")
    
    return a.view(-1, 1)


def metricas(figures_dir, results_dir ,modelo_dir, h5file, idx_dir):
    """
    Gera gráficos a partir dos resultados salvos
    """
    # Carrega os dados
    hs, Y, tam_Y, indices_validos = load_hs_fp_dataset(h5file)
    
    # Carrega os índices
    train_idx, val_idx = load_shared_idx(idx_dir)
    
    # Carrega as previsões
    all_pred = np.load(os.path.join(modelo_dir, "all_pred.npy"))
    all_true = np.load(os.path.join(modelo_dir, "all_true.npy"))
    
    # Converte para escala original
    all_pred_exp = 10 ** all_pred.flatten()
    all_true_exp = 10 ** all_true.flatten()
    
    
    # ============ GRÁFICO 1: q vs Hs ============

    a_val = 10 ** all_pred  
    q_val = a_val / (hs[val_idx] ** 6)

    model_params_path = os.path.join(modelo_dir, "amplitude_model_params.pth")
    if os.path.exists(model_params_path):
        checkpoint = torch.load(model_params_path, map_location='cpu')
        coef = checkpoint['coef_'].numpy()[0]
        intercept = checkpoint['intercept_'].numpy()
        print(f"Coeficiente carregado: w1 = {coef:.6f}")
        print(f"Intercept carregado: b = {intercept:.6f}")

    
    plt.figure(figsize=(6, 5))
    plt.scatter(hs[val_idx], q_val, alpha=0.3, s=5)
    plt.xlabel('Hs (m)')
    plt.ylabel('a / Hs^6')
    plt.title(f'q pred vs Hs - {len(q_val)} amostras de validação')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "q_pred_vs_hs.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo em: {os.path.join(figures_dir, 'q_pred_vs_hs.pdf')}")


    
    # ============ GRÁFICO 2: log(a_pred) vs log(Hs) ============
    plt.figure(figsize=(6, 5))
    plt.scatter(np.log10(hs[val_idx]), all_true, alpha=0.3, s=5, label='Real')
    plt.scatter(np.log10(hs[val_idx]), all_pred, alpha=0.3, s=5, label='Predito')
    plt.xlabel('log10(Hs)')
    plt.ylabel('log10(a)')
    plt.title(f'log(a) vs log(Hs) - {len(all_true)} amostras de validação - B1={coef:.6f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "log_amplitude_vs_log_hs.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo em: {os.path.join(figures_dir, 'log_amplitude_vs_log_hs.pdf')}")
    
 
    # ============ TABELA COM DADOS ============
    # Cria tabela com os dados
    hs_np = hs[val_idx].numpy().flatten() if hasattr(hs[val_idx], 'numpy') else hs[val_idx].flatten()
    log_hs = np.log10(hs_np)
    a_true = 10 ** all_true.flatten()
    a_pred = 10 ** all_pred.flatten()
    log_a_true = all_true.flatten()
    log_a_pred = all_pred.flatten()

    
    # Cria DataFrame
    df = pd.DataFrame({
        'Hs (m)': hs_np,
        'log10(Hs)': log_hs,
        'a (true)': a_true,
        'a (pred)': a_pred,
        'log10(a_true)': log_a_true,
        'log10(a_pred)': log_a_pred,
        'a / Hs^6': q_val
    })
    
    # Salva como CSV
    csv_path = os.path.join( results_dir, "dados_validacao.csv")
    df.to_csv(csv_path, index=False, float_format='%.6e')
    print(f"Tabela salva em: {csv_path}")
    
    # Salva como TXT (formato mais legível)
    txt_path = os.path.join( results_dir, "dados_validacao.txt")
    with open(txt_path, 'w') as f:
        f.write("="*120 + "\n")
        f.write("TABELA DE DADOS - VALIDAÇÃO\n")
        f.write("="*120 + "\n\n")
        f.write(f"{'Hs (m)':>12} {'log10(Hs)':>12} {'a (true)':>18} {'a (pred)':>18} {'log10(a_true)':>14} {'log10(a_pred)':>14} {'a / Hs^6':>18}\n")
        f.write("-"*120 + "\n")
        for i in range(len(hs_np)):
            f.write(f"{hs_np[i]:>12.4f} {log_hs[i]:>12.4f} {a_true[i]:>18.6e} {a_pred[i]:>18.6e} {log_a_true[i]:>14.4f} {log_a_pred[i]:>14.4f} {q_val[i]:>18.6e}\n")
        f.write("-"*120 + "\n")
        f.write(f"\nTotal de amostras: {len(hs_np)}\n")
        f.write(f"\nCoeficiente do modelo: w1 = {coef:.6f}\n")
        f.write(f"Intercept do modelo: b = {intercept:.6f}\n")
        f.write(f"Equação: log10(a) = {coef:.6f} * log10(Hs) + {intercept:.6f}\n")
    print(f"Tabela salva em: {txt_path}")

    print("\n========== DEBUG Q ==========")

    print("all_pred.shape:", all_pred.shape)

    print("val_idx.shape:", val_idx.shape)

    print("Hs[val_idx].shape:", hs[val_idx].shape)

    print("Hs mínimo:", hs[val_idx].min().item())
    print("Hs máximo:", hs[val_idx].max().item())

    print("all_pred mínimo:", all_pred.min())
    print("all_pred máximo:", all_pred.max())

    a_val = 10 ** all_pred.flatten()

    hs_val = hs[val_idx].numpy().flatten()

    q_val = a_val / (hs_val ** 6)

    print("a_val mínimo:", a_val.min())
    print("a_val máximo:", a_val.max())

    print("q mínimo:", q_val.min())
    print("q máximo:", q_val.max())

    print("==============================\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5file", type=str, help="Arquivo H5 com os dados", default=DATA_PATH)
    parser.add_argument("--dir_out", type=str, help="Diretório com os resultados", default=FIG_DIR)
    parser.add_argument("--dir_out_table", type=str, help="Diretório com os resultados", default=RES_DIR)
    parser.add_argument("--dir_modelo", type=str, help="Diretório com as metricas", default=MODELO_DIR)
    parser.add_argument("--idx_dir", type=str, help="Diretório com os índices", default=IDX_DIR)
    args = parser.parse_args()
    
    metricas(args.dir_out,args.dir_out_table,args.dir_modelo, args.h5file, args.idx_dir)