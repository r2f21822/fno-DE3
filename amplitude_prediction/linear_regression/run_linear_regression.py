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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score,mean_absolute_percentage_error
from sklearn.model_selection import train_test_split


DATA_PATH = "Generate_data/snl/snl_dataset.h5"
RUN_DIR = "amplitude_prediction/linear_regression/results"
#depois retirar daqui e colocar em outra pasta
FIG_DIR ="amplitude_prediction/linear_regression/figures"

IDX_DIR="train_test_split/results"



def load_shared_idx(idx_dir=IDX_DIR):

    train_idx = np.load(os.path.join(idx_dir, "train_indices.npy"))
    val_idx = np.load(os.path.join(idx_dir, "val_indices.npy"))
    
    return train_idx, val_idx

def factorize_target(Y):
    """
    Y: (N, 1, Nf, Ntheta)
    Retorna:
      a: max_abs(Y), shape (N, 1)
    """
    #eve verificar se a[i] e finito e maior que zero. Se aparecer uma amostra invalida, o programa deve parar e mostrar uma mensagem clara
    a = Y.abs().amax(dim=(1, 2, 3), keepdim=True) 
    for i, val in enumerate(a.view(-1)):
        #infinito ou nan
        if not torch.isfinite(val):
            raise ValueError(f"Amostra {i} inválida: amplitude não é finita (valor: {val})")
        if val <= 0:
           raise ValueError(f"Amostra {i} inválida: amplitude menor que zero (valor: {val})")
            
    shape = Y / a 
    
    for i in range(len(shape)):
        max_abs = shape[i].abs().max().item()
        if not (max_abs==1):
            raise ValueError(f"Amostra {i} inválida: max(abs(shape)) = {max_abs}, deveria ser 1")
    
        
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
    
    tam_Y=len(Y)
    
    return Hs, fp,Y, gamma,s, tam_Y



def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--h5file", default=DATA_PATH)
    pa.add_argument("--out_dir", default=RUN_DIR)
    pa.add_argument("--out_dir_figs", default=FIG_DIR)
    pa.add_argument("--idx_dir", default=IDX_DIR)

    args = pa.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.out_dir_figs, exist_ok=True)
 



    hs, fp, Y, gamma, s,tam_Y = load_hs_fp_dataset(args.h5file)
    
    a = factorize_target(Y)
    a_log = np.log10(a)
    
    indices = np.arange(tam_Y)
    
  
    X = np.column_stack([
        np.log10(hs),
        np.log10(fp),
        np.log10(gamma),
        s
    ])
    


    train_idx, val_idx=load_shared_idx(args.idx_dir)
    
    X_train = X[train_idx]
    X_val = X[val_idx]
    y_train = a_log[train_idx].flatten()
    y_val = a_log[val_idx].flatten()
    

    model = LinearRegression()
    model.fit(X_train, y_train)
    
 
    all_pred = model.predict(X_val)

    mae_log = mean_absolute_error(y_val, all_pred)
    mse_log=mean_squared_error(y_val, all_pred)
    rmse_log = np.sqrt(mean_squared_error(y_val, all_pred))
    r2_log = r2_score(y_val, all_pred)
    

    y_val_exp = 10 ** y_val
    all_pred_exp = 10 ** all_pred
    
    mae_original = mean_absolute_error(y_val_exp, all_pred_exp)
    mse_original= mean_squared_error(y_val_exp, all_pred_exp)
    rmse_original = np.sqrt(mean_squared_error(y_val_exp, all_pred_exp))
    mape_original = mean_absolute_percentage_error(y_val_exp, all_pred_exp)
    r2_original = r2_score(y_val_exp, all_pred_exp)
    
    
  
    # Salvar resultados
    os.makedirs(RUN_DIR, exist_ok=True)
    np.save(os.path.join(RUN_DIR, "all_pred.npy"), all_pred)
    np.save(os.path.join(RUN_DIR, "all_true.npy"), y_val)


    print(f"Previsões salvas em: {args.out_dir}")


    #para utilizar como dados nos grafico
    model_params = {
        'coef_': torch.tensor(model.coef_, dtype=torch.float32),
        'intercept_': torch.tensor(model.intercept_, dtype=torch.float32),
        'feature_names': ['log10(Hs)', 'log10(fp)', 'log10(gamma)', 's'],
        'model_type': 'LinearRegression',
        'n_features': X_train.shape[1],
        
    }
    

    torch.save(model_params, os.path.join(args.out_dir, 'amplitude_model_params.pth'))
    print(f"Parâmetros .pth salvos: {os.path.join(args.out_dir, 'amplitude_model_params.pth')}")

    #para utilizar em um yamal para poder visualizar os resultados, futuramente juntar ambos e modificar os ourtros arquivos afetadps 
    results = {
        'coefficients': {
            'intercept': float(model.intercept_),           
            'log10_hs': float(model.coef_[0]),            
            'log10_fp': float(model.coef_[1]),             
            'log10_gamma': float(model.coef_[2]),          
            's': float(model.coef_[3]),                    
        },
        'metrics': {
            'MAE (escala original)': float(mae_original), 
            'MSE (escala original)': float(mse_original), 
            'RMSE (escala original)': float(rmse_original), 
            'MAPE (escala original)': float(mape_original),
            'R2 (escala original)': float(r2_original),
            'MAE (escala log)': float(mae_log), 
            'MSE (escala log)': float(mse_log), 
            'RMSE (escala log)': float(rmse_log),            
            'R2 (escala log)': float(r2_log),                            

        }
    }
    
    
  
    config_treino={
    'Configuracao do conjunto de amostras':{
            'Tamanho do conjunto de amostras total': int(len(Y)),
            'Tamanho do conjunto de treino': int(len(X_train)),
            'Tamanho do conjunto de validacao':int(len(X_val)),
            'Porcentagem para treino': float(len(X_train)/int(len(Y))),
            'Porcentagem para validacao': float(len(X_val)/int(len(Y))),
            'Indices treino': train_idx.tolist(),  # Para conferência
            'Indices validacao': val_idx.tolist(),  # Para conferência
        }   
    
    }
    
    with open(os.path.join(RUN_DIR, "amplitude_model_metrics.yaml"), "w") as f:
        yaml.dump(results, f, default_flow_style=False,sort_keys=False)
        
    with open(os.path.join(RUN_DIR, "amplitude_model_configuracoes_treino.yaml"), "w") as f:
        yaml.dump(config_treino, f, default_flow_style=False,sort_keys=False)

    #grafico de log(A)

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
    plt.title(f'Real vs Predito - {len(y_val)} amostras de validação')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir_figs, "amplitude__logtrue_vs_logpred.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"grafico log(A_true) vs log(A_pred) salvo em: {os.path.join(args.out_dir_figs, 'amplitude__logtrue_vs_logpred.pdf')}")

    print(f"\n Treinamento finalizado")
    print(f" Arquivos salvos em: {args.out_dir}")


    print("Coeficientes encontrados:")
    print(f"  w1 = {model.coef_[0]:.6f}  (log(Hs))")
    print(f"  w2 = {model.coef_[1]:.6f}  (log(fp))")
    print(f"  w3 = {model.coef_[2]:.6f}  (log(gamma))")
    print(f"  w4 = {model.coef_[3]:.6f}  (s)")
    print(f"  b  = {model.intercept_:.6f}")
    

    print("\n" + "="*60)
    print("MÉTRICAS EM log10(A)")
    print("="*60)
    print(f" log(A) MAE:  {mae_log:.6f}")
    print(f" log(A) MSE:  {mse_log:.6f}")
    print(f" log(A) RMSE: {rmse_log:.6f}")
    print(f" log(A)  R²:   {r2_log:.6f}")
    print("\n" + "="*60)
    print("MÉTRICAS EM A (escala original)")
    print("="*60)
    print(f" A MAE:  {mae_original:.6f}")
    print(f" A MSE:  {mse_original:.6f}")
    print(f" A RMSE: {rmse_original:.6f}")
    print(f" A R2:   {r2_original:.6f}")
    print("="*60)


    min_val = min( y_val_exp.min(), all_pred_exp.min())
    max_val = max(y_val_exp.max(), all_pred_exp.max())

    plt.figure(figsize=(6, 5))
    plt.scatter(y_val_exp, all_pred_exp, alpha=0.3, s=5)

    plt.xlim(min_val - 0.5, max_val + 0.5)
    plt.ylim(min_val - 0.5, max_val + 0.5)
    plt.axis('equal')


    plt.plot([min_val - 0.5, max_val + 0.5],
             [min_val - 0.5, max_val + 0.5],
             'r--', lw=2, label='Ideal')

    plt.xlabel('Amplitude Real')
    plt.ylabel('Amplitude Predita')
    plt.title(f'Real vs Predito - {len(y_val)} amostras de validação')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir_figs, "amplitude_original_true_vs_pred.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"grafico de amplitude_rel vs amplitude_predita salvo em: {os.path.join(args.out_dir_figs, 'amplitude_original_true_vs_pred.pdf')}")

  





if __name__ == "__main__":
    main()