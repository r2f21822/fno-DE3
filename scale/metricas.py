import numpy as np
import matplotlib.pyplot as plt
import os

def metricas(results_dir):
    
    all_pred = np.load(os.path.join(results_dir, "all_pred.npy"))
    all_true = np.load(os.path.join(results_dir, "all_true.npy"))
   
    
  
    all_pred_exp = 10 ** all_pred
    all_true_exp = 10 ** all_true
    

    print("log(A) - Erro Medio Absoluto: ",np.mean(np.abs(all_true - all_pred)))  # erro médio absoluto
    print("log(A) - Erro Quadratico Médio :",np.mean((all_true - all_pred) ** 2))   
    print("log(A) - Raiz Erro Quadratico Medio",np.sqrt(np.mean((all_true - all_pred) ** 2)))             


    print("A - Erro Medio Absoluto: ",np.mean(np.abs(all_true_exp - all_pred_exp)))  
    print("A - Erro Quadratico Médio :",np.mean((all_true_exp - all_pred_exp) ** 2))   
    print("A - Raiz Erro Quadratico Medio",np.sqrt(np.mean((all_true_exp - all_pred_exp) ** 2))) 
    
    rmse_percentual = (np.sqrt(np.mean((all_true_exp - all_pred_exp) ** 2)) / np.mean(all_true_exp)) * 100
    print(f"RMSE percentual: {rmse_percentual}%")


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
    mask_menor = all_true_exp <= mediana_real
    true_menor = all_true_exp[mask_menor]
    pred_menor = all_pred_exp[mask_menor]

    n_menor = len(true_menor)
    mae_menor = np.mean(np.abs(true_menor - pred_menor))
    rmse_menor = np.sqrt(np.mean((true_menor - pred_menor) ** 2))
    mape_menor = np.mean(np.abs((true_menor - pred_menor) / (true_menor + 1e-8))) * 100


    print("\n" + "-"*60)
    print("METADE MENOR (amplitudes <= mediana)")
    print(f"Número de amostras:",n_menor)
    print(f"Faixa de amplitude: {true_menor.min():.6f} a {true_menor.max():.6f}")
    print(f"A - Erro Medio Absoluto:        {mae_menor:.6f}")
    print(f"A - Erro Quadratico Médio :     {rmse_menor:.6f}")
    print(f"A - Raiz Erro Quadratico Medio: {mape_menor:.2f}%")

    # METADE MAIOR (amplitudes > mediana)


    mask_maior = all_true_exp > mediana_real
    true_maior = all_true_exp[mask_maior]
    pred_maior = all_pred_exp[mask_maior]

    n_maior = len(true_maior)
    mae_maior = np.mean(np.abs(true_maior - pred_maior))
    rmse_maior = np.sqrt(np.mean((true_maior - pred_maior) ** 2))
    mape_maior = np.mean(np.abs((true_maior - pred_maior) / (true_maior + 1e-8))) * 100

    print("\n" + "-"*60)
    print("METADE MAIOR (amplitudes > mediana)")
    print(f"Número de amostras:",n_maior)
    print(f"Faixa de amplitude: {true_maior.min():.6f} a {true_maior.max():.6f}")
    print(f"A - Erro Medio Absoluto:        {mae_maior:.6f}")
    print(f"A - Erro Quadratico Médio :     {rmse_maior:.6f}")
    print(f"A - Raiz Erro Quadratico Medio: {mape_maior:.2f}%")



   

if __name__ == "__main__":
    RESULTS_DIR = "scale/results_scale_loggamma_s"
    metricas(RESULTS_DIR)