import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

RUN_DIR="amplitude_prediction/linear_regression/results"

def metricas(results_dir):
    
    all_pred = np.load(os.path.join(results_dir, "all_pred.npy"))
    all_true = np.load(os.path.join(results_dir, "all_true.npy"))

    
  
    all_pred_exp = 10 ** all_pred
    all_true_exp = 10 ** all_true

    #METRICAS COM LOG (A)
    
    mae = np.mean(np.abs(all_true - all_pred))
    mse = np.mean((all_true - all_pred) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((all_true - all_pred) / (all_true))) * 100
    rmse_percentual = (rmse / np.mean(all_true)) * 100
    
    print(f"log (A): MAE  (Erro Absoluto Médio):              {mae:.6f}")
    print(f"log (A): MSE  (Erro Quadrático Médio):            {mse:.6f}")
    print(f"log (A): RMSE (Raiz do Erro Quadrático Médio):    {rmse:.6f}")
   


    # METRICAS COM A AMPLITUDE REAL
    print("\n" + "-"*60 + "\n")
    mae = np.mean(np.abs(all_true_exp - all_pred_exp))
    mse = np.mean((all_true_exp - all_pred_exp) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((all_true_exp - all_pred_exp) / (all_true_exp))) * 100
    rmse_percentual = (rmse / np.mean(all_true_exp)) * 100
    
    print(f"A: MAE  (Erro Absoluto Médio):              {mae:.6f}")
    print(f"A: MSE  (Erro Quadrático Médio):            {mse:.6f}")
    print(f"A: RMSE (Raiz do Erro Quadrático Médio):    {rmse:.6f}")
    print(f"A: MAPE (Erro Percentual Absoluto Médio):   {mape:.2f}%")



#METRICAS COM AS MEDIANAS

    print("\n" + "-"*60 + "\n")
    mediana_real = np.median(all_true_exp)
    media_real = np.mean(all_true_exp)


    print(f"Mediana das amplitudes reais: ", mediana_real)
    print(f"Média das amplitudes reais: ", media_real)

    
    mediana_pred= np.median(all_pred_exp)
    media_pred = np.mean(all_pred_exp)


    print(f"Mediana das amplitudes preditas: ", mediana_pred)
    print(f"Média das amplitudes preditas: ", media_pred)


    print("Metricas olhando metade () das amostras")

    #METADE MENOR (amplitude<=mediana)

    print("\n" + "-"*60 + "\n")
    mask_menor = all_true_exp <= mediana_real
    true_menor = all_true_exp[mask_menor]
    pred_menor = all_pred_exp[mask_menor]
    
    n_menor = len(true_menor)
    mae_menor = np.mean(np.abs(true_menor - pred_menor))
    rmse_menor = np.sqrt(np.mean((true_menor - pred_menor) ** 2))
    mape_menor = np.mean(np.abs((true_menor - pred_menor) / (true_menor))) * 100
    
    print("\n METADE MENOR (amplitudes <= mediana)")
    print(f"  Número de amostras:        {n_menor}")
    print(f"  Faixa de amplitude:        {true_menor.min():.6f} a {true_menor.max():.6f}")
    print(f"  MAE  (Erro Absoluto Médio):        {mae_menor:.6f}")
    print(f"  RMSE (Raiz do Erro Quadrático):    {rmse_menor:.6f}")
    print(f"  MAPE (Erro Percentual Médio):      {mape_menor:.2f}%")


    # METADE MAIOR (amplitudes > mediana)

    print("\n" + "-"*60 + "\n")
    mask_maior = all_true_exp > mediana_real
    true_maior = all_true_exp[mask_maior]
    pred_maior = all_pred_exp[mask_maior]
    
    n_maior = len(true_maior)
    mae_maior = np.mean(np.abs(true_maior - pred_maior))
    rmse_maior = np.sqrt(np.mean((true_maior - pred_maior) ** 2))
    mape_maior = np.mean(np.abs((true_maior - pred_maior) / (true_maior))) * 100
    
    print("\n METADE MAIOR (amplitudes > mediana)")
    print(f"  Número de amostras:        {n_maior}")
    print(f"  Faixa de amplitude:        {true_maior.min():.6f} a {true_maior.max():.6f}")
    print(f"  MAE  (Erro Absoluto Médio):        {mae_maior:.6f}")
    print(f"  RMSE (Raiz do Erro Quadrático):    {rmse_maior:.6f}")
    print(f"  MAPE (Erro Percentual Médio):      {mape_maior:.2f}%")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Diretório com os resultados",default=RUN_DIR)
    args = parser.parse_args()
    
    metricas(args.dir)