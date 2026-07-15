import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
import os
import argparse

MODELO_DIR = "amplitude_prediction/linear_regression/results"
#onde vai ser salvo
RUN_DIR="amplitude_prediction/linear_regression/figures/out"

def metricas(results_dir,modelo_dir):
   
    all_pred = np.load(os.path.join(modelo_dir, "all_pred.npy"))
    all_true = np.load(os.path.join(modelo_dir, "all_true.npy"))
    
   
    all_pred_exp = 10 ** all_pred.flatten()
    all_true_exp = 10 ** all_true.flatten()
    
   
    erro_percentual = ((all_true_exp - all_pred_exp) / (all_true_exp + 1e-10)) * 100
    erro_absoluto = np.abs(all_true_exp - all_pred_exp)
    mape = np.mean(erro_percentual)
    mediana_erro = np.median(erro_percentual)
    std_erro = np.std(erro_percentual)
    
    mae = np.mean(erro_absoluto)
    mediana_erro_abs = np.median(erro_absoluto)
    std_erro_abs = np.std(erro_absoluto)
    print("="*60)
    print("ESTATÍSTICAS DO ERRO PERCENTUAL")
    print("="*60)
    print(f"MAPE (Média):     {mape:.2f}%")
    print(f"Mediana:          {mediana_erro:.2f}%")
    print(f"Desvio padrão:    {std_erro:.2f}%")
    print(f"Mínimo:           {erro_percentual.min():.2f}%")
    print(f"Máximo:           {erro_percentual.max():.2f}%")
    print(f"Q1 (25%):         {np.percentile(erro_percentual, 25):.2f}%")
    print(f"Q3 (75%):         {np.percentile(erro_percentual, 75):.2f}%")
    print("="*60)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # PLOT 1: Erro percentual e Amplitude correspondente ===================================================

    sorted_idx_amp = np.argsort(all_true_exp)
    erro_sorted = erro_percentual[sorted_idx_amp]
    amp_sorted = all_true_exp[sorted_idx_amp]

    # Plotar cada amostra como um ponto
    scatter = ax1.scatter(np.arange(len(erro_sorted)), erro_sorted, 
                        alpha=0.6, s=20, c=erro_sorted, 
                        cmap='plasma', edgecolors='black', linewidth=0.5)

    # Linhas de referência
    ax1.axhline(y=mape, color='red', linestyle='--', linewidth=2, 
                label=f'Média (MAPE): {mape:.2f}%')
    ax1.axhline(y=mediana_erro, color='green', linestyle='--', linewidth=2, 
                label=f'Mediana: {mediana_erro:.2f}%')

    ax1.set_xlabel("Amplitude Real", fontsize=12)
    ax1.set_ylabel("Erro Percentual  (%)", fontsize=12)
    ax1.set_title(f'Erro Percentual por Amostra\n{len(all_true_exp)} amostras', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Adicionar barra de cores
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Erro Percentual (%)', fontsize=10)

    # Adicionar segundo eixo X com a amplitude
   
    # Mostrar algumas amplitudes no eixo superior
    n_ticks = 5
    tick_positions = np.linspace(0, len(erro_sorted)-1, n_ticks, dtype=int)
    tick_labels = [f'$10^{{{int(np.log10(amp_sorted[i]))}}}$' for i in tick_positions]
    ax1.set_xticks(tick_positions)
    ax1.set_xticklabels(tick_labels, fontsize=8)

    erro_percentual_absoluto = np.abs(erro_percentual)

    mape_abs = np.mean(erro_percentual_absoluto)  
    mediana_erro_abs = np.median(erro_percentual_absoluto)  
    std_erro_abs = np.std(erro_percentual_absoluto) 


    
        # PLOT 2: Erro percentual absoluto ==============================================
    erro_abs_sorted = erro_percentual_absoluto[sorted_idx_amp]

    scatter2 = ax2.scatter(np.arange(len(erro_abs_sorted)), erro_abs_sorted, 
                        alpha=0.6, s=20, c=erro_abs_sorted, 
                        cmap='viridis', edgecolors='black', linewidth=0.5)

    ax2.axhline(y=mape_abs, color='red', linestyle='--', linewidth=2, 
                label=f'MAPE Abs: {mape_abs:.2f}%')
    ax2.axhline(y=mediana_erro_abs, color='green', linestyle='--', linewidth=2, 
                label=f'Mediana: {mediana_erro_abs:.2f}%')

    ax2.set_xlabel("Amplitude Real", fontsize=12)
    ax2.set_ylabel("Erro Percentual Absoluto (%)", fontsize=12)
    ax2.set_title(f'Erro Percentual Absoluto por Amostra\n{len(all_true_exp)} amostras', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')

    cbar2 = plt.colorbar(scatter2, ax=ax2)
    cbar2.set_label('Erro Percentual Absoluto (%)', fontsize=10)

    # Eixo superior para o segundo gráfico
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(tick_labels, fontsize=8)



    out_file = os.path.join(results_dir, "erro_percentual_por_amostra_ordenada.pdf")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo em: {out_file}")
    
    
    return {
        'mape': mape,
        'mediana': mediana_erro,
        'std': std_erro,
        'min': erro_percentual.min(),
        'max': erro_percentual.max(),
        'q1': np.percentile(erro_percentual, 25),
        'q3': np.percentile(erro_percentual, 75)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_out", type=str, help="Diretório com os resultados", default=RUN_DIR)
    parser.add_argument("--dir_modelo", type=str, help="Diretório com as metricas",default=MODELO_DIR)
    args = parser.parse_args()
    
    metricas(args.dir_out, args.dir_modelo)