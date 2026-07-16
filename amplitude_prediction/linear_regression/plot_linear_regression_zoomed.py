import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import os
import argparse


MODELO_DIR = "amplitude_prediction/linear_regression/results"
#onde vai ser salvo
RUN_DIR="amplitude_prediction/linear_regression/figures"

def plot_linear_regression(results_dir,modelo_dir):


    all_pred = np.load(os.path.join(modelo_dir, "all_pred.npy"))
    all_true = np.load(os.path.join(modelo_dir, "all_true.npy"))
    
   
    all_pred_exp = 10 ** all_pred.flatten()
    y_val_exp = 10 ** all_true.flatten()

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
        plt.title(f'Real vs Predito (escala original) - {len(y_val_exp_filtrado)} amostras de validação')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        nome_grafico_limitado = "amplitude_original_limitado_ate" + str(limitacao) + ".pdf"

        plt.savefig(os.path.join(args.out_dir_figs, nome_grafico_limitado), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"grafico limitado (escala original) salvo em: {os.path.join(args.out_dir_figs, nome_grafico_limitado)}")


    
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
        plt.title(f'Real vs Predito (escala original) - {len(y_val_exp_filtrado)} amostras de validação')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()


        nome_grafico_limitado = "amplitude_original_limitado_ate" + str(min_amostras) + "amostras.pdf"

        plt.savefig(os.path.join(args.out_dir_figs, nome_grafico_limitado), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"grafico limitado (escala original) salvo em: {os.path.join(args.out_dir_figs, nome_grafico_limitado)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir_figs", type=str, help="Diretório com os grafics", default=RUN_DIR)
    parser.add_argument("--dir_modelo", type=str, help="Diretório com o modelo de amplitude",default=MODELO_DIR)
    args = parser.parse_args()
    
    plot_linear_regression(args.out_dir_figs, args.dir_modelo)