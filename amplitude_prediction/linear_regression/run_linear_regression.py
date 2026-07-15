#log de gamma e s normal
#arquivo principal
#mudando para sklearn


import argparse
import json
import os
import yaml

import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchmetrics.regression import R2Score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


DATA_PATH = "Generate_data/snl/snl_dataset.h5"
RUN_DIR = "amplitude_prediction/linear_regression/results"
#depois retirar daqui e colocar em outra pasta
FIG_DIR ="amplitude_prediction/linear_regression/figures/out"
EPS = 1e-8




def factorize_target(Y, eps=EPS):
    """
    Y: (N, 1, Nf, Ntheta)
    Retorna:
      a: max_abs(Y), shape (N, 1)
    """
    a = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + eps
    return a.view(-1, 1)


def load_hs_fp_dataset(path):
   
    with h5py.File(path, "r") as hf:
        Hs = hf["Hs"][:]   
        fp = hf["fp"][:]   
        gamma = hf["gamma"][:]   
        s = hf["s"][:]   
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


def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--h5file", default=DATA_PATH)
    pa.add_argument("--out-dir", default=RUN_DIR)
    pa.add_argument("--out-dir_figs", default=FIG_DIR)
    pa.add_argument("--epochs", type=int, default=100)
    pa.add_argument("--batch", type=int, default=16)
    pa.add_argument("--lr", type=float, default=1e-3)
    pa.add_argument("--scale-head-hidden", type=int, default=128)
    pa.add_argument("--seed", type=int, default=42)
    args = pa.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_dir_figs, exist_ok=True)



    hs, fp, Y, gamma, s = load_hs_fp_dataset(args.h5file)
    
    a = factorize_target(Y)
    a_log = np.log10(a)
    
  
    X = np.column_stack([
        np.log10(hs + EPS),
        np.log10(fp + EPS),
        np.log10(gamma),
        s
    ])
    
   
    X_train, X_val, y_train, y_val = train_test_split(
        X, a_log, test_size=0.2, random_state=42
    )

    y_train = np.array(y_train).flatten()  
    y_val = np.array(y_val).flatten()
    

    model = LinearRegression()
    model.fit(X_train, y_train)
    

    print("Coeficientes encontrados:")
    print(f"  w1 = {model.coef_[0]:.6f}  (log(Hs))")
    print(f"  w2 = {model.coef_[1]:.6f}  (log(fp))")
    print(f"  w3 = {model.coef_[2]:.6f}  (log(gamma))")
    print(f"  w4 = {model.coef_[3]:.6f}  (s)")
    print(f"  b  = {model.intercept_:.6f}")
    
 
    all_pred = model.predict(X_val)

    mae_log = mean_absolute_error(y_val, all_pred)
    rmse_log = np.sqrt(mean_squared_error(y_val, all_pred))
    print(f"\nlog(A) - MAE: {mae_log:.6f}")
    print(f"log(A) - RMSE: {rmse_log:.6f}")
    

    y_val_exp = 10 ** y_val
    all_pred_exp = 10 ** all_pred
    
    mae = mean_absolute_error(y_val_exp, all_pred_exp)
    rmse = np.sqrt(mean_squared_error(y_val_exp, all_pred_exp))
    r2 = r2_score(y_val_exp, all_pred_exp)
    
    print(f"\nA - MAE: {mae:.6f}")
    print(f"A - RMSE: {rmse:.6f}")
    print(f"A - R²: {r2:.6f}")
    
    # Salvar resultados
    os.makedirs(RUN_DIR, exist_ok=True)
    np.save(os.path.join(RUN_DIR, "all_pred.npy"), all_pred)
    np.save(os.path.join(RUN_DIR, "all_true.npy"), y_val)


    print(f"Previsões salvas em: {args.out_dir}")

    min_val = min(y_val.min(), all_pred.min())
    max_val = max(y_val.max(), all_pred.max())

    plt.figure(figsize=(6, 5))
    plt.scatter(y_val, all_pred, alpha=0.3, s=5)

    plt.xlim(min_val - 0.5, max_val + 0.5)
    plt.ylim(min_val - 0.5, max_val + 0.5)
    plt.axis('equal')


    plt.plot([min_val - 0.5, max_val + 0.5],
             [min_val - 0.5, max_val + 0.5],
             'r--', lw=2, label='Ideal')

    plt.xlabel('log(Amplitude Real)')
    plt.ylabel('log(Amplitude Predita)')
    plt.title(f'Real vs Predito - {len(y_val)} amostras')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir_figs, "amplitude_emLog_loggamma_s.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Scatter salvo em: {os.path.join(args.out_dir_figs, 'amplitude_emLog_loggamma_s.pdf')}")

    print(f"\n Treinamento finalizado")
    print(f" Arquivos salvos em: {args.out_dir}")

    print("\n" + "="*60)
    print("MÉTRICAS EM log10(A)")
    print("="*60)
    mae_log = mean_absolute_error(y_val, all_pred)
    mse_log = mean_squared_error(y_val, all_pred)
    rmse_log = np.sqrt(mse_log)
    print(f"  MAE:  {mae_log:.6f}")
    print(f"  MSE:  {mse_log:.6f}")
    print(f"  RMSE: {rmse_log:.6f}")

    
    # ========== MÉTRICAS NA ESCALA ORIGINAL ==========
    y_val_exp = 10 ** y_val
    all_pred_exp = 10 ** all_pred
    
    print("\n" + "="*60)
    print("MÉTRICAS EM A (escala original)")
    print("="*60)
    mae = mean_absolute_error(y_val_exp, all_pred_exp)
    mse = mean_squared_error(y_val_exp, all_pred_exp)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val_exp, all_pred_exp)
    print(f"  MAE:  {mae:.6f}")
    print(f"  MSE:  {mse:.6f}")
    print(f"  RMSE: {rmse:.6f}")
    print(f"  R²:   {r2:.6f}")
    print("="*60)



    listaLimitaoes=[np.max(y_val_exp),1,0.1,0.001,0.0001]
    
    for limitacao in listaLimitaoes:
        mask = (y_val_exp >= 0) & (y_val_exp <= limitacao) & (all_pred_exp >= 0) & (all_pred_exp <= limitacao)
        y_val_exp_filtrado = y_val_exp[mask]
        all_pred_exp_filtrado = all_pred_exp[mask]

        min_val_exp = min(y_val_exp_filtrado.min(), all_pred_exp_filtrado.min())
        max_val_exp = max(y_val_exp_filtrado.max(), all_pred_exp_filtrado.max())

        plt.figure(figsize=(6, 5))
        plt.scatter(y_val_exp_filtrado, all_pred_exp_filtrado, alpha=0.3, s=5)

        plt.xlim(min_val_exp * 0.9, max_val_exp * 1.1)
        plt.ylim(min_val_exp * 0.9, max_val_exp * 1.1)

        plt.axis('equal')

        plt.plot([min_val_exp * 0.9, max_val_exp * 1.1],
                [min_val_exp * 0.9, max_val_exp * 1.1],
                'r--', lw=2, label='Ideal')

        plt.xlabel('Amplitude Real (10^x)')
        plt.ylabel('Amplitude Predita (10^x)')
        plt.title(f'Real vs Predito (escala original) - {len(y_val_exp_filtrado)} amostras')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        nome_grafico_limitado = "amplitude_original_limitado_ate" + str(limitacao) + ".pdf"

        plt.savefig(os.path.join(args.out_dir_figs, nome_grafico_limitado), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Scatter (escala original) salvo em: {os.path.join(args.out_dir_figs, nome_grafico_limitado)}")


    
    num_amostras=len(y_val_exp)
    listaLimitaoesPorAmostra = [
        int(num_amostras),           
        int(num_amostras * 0.75),    
        int(num_amostras // 2),      
        int(num_amostras // 4)       
    ]
    
    for min_amostras in listaLimitaoesPorAmostra:
        
        valores_ordenados = np.sort(y_val_exp)
    
        limite_amplitude = valores_ordenados[min_amostras - 1]

        mask = (y_val_exp >= 0) & (y_val_exp <= limite_amplitude) & (all_pred_exp >= 0) & (all_pred_exp <= limite_amplitude)
        y_val_exp_filtrado = y_val_exp[mask]
        all_pred_exp_filtrado = all_pred_exp[mask]

        min_val_exp = min(y_val_exp_filtrado.min(), all_pred_exp_filtrado.min())
        max_val_exp = max(y_val_exp_filtrado.max(), all_pred_exp_filtrado.max())

        plt.figure(figsize=(6, 5))
        plt.scatter(y_val_exp_filtrado, all_pred_exp_filtrado, alpha=0.3, s=5)

        plt.xlim(min_val_exp * 0.9, max_val_exp * 1.1)
        plt.ylim(min_val_exp * 0.9, max_val_exp * 1.1)

        plt.axis('equal')

        plt.plot([min_val_exp * 0.9, max_val_exp * 1.1],
                [min_val_exp * 0.9, max_val_exp * 1.1],
                'r--', lw=2, label='Ideal')

        plt.xlabel('Amplitude Real (10^x)')
        plt.ylabel('Amplitude Predita (10^x)')
        plt.title(f'Real vs Predito (escala original) - {len(y_val_exp_filtrado)} amostras')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        nome_grafico_limitado = "amplitude_original_limitado_ate" + str(min_amostras) + "amostras.pdf"

        plt.savefig(os.path.join(args.out_dir_figs, nome_grafico_limitado), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Scatter (escala original) salvo em: {os.path.join(args.out_dir_figs, nome_grafico_limitado)}")
        

if __name__ == "__main__":
    main()