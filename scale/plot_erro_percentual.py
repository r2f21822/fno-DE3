import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

MODELO_DIR = "scale/results_scale_loggamma_s"
#onde vai ser salvo
RUN_DIR="scale/results_scale_loggamma_s"

def metricas(results_dir,modelo_dir):
   
    all_pred = np.load(os.path.join(modelo_dir, "all_pred.npy"))
    all_true = np.load(os.path.join(modelo_dir, "all_true.npy"))
    
   
    all_pred_exp = 10 ** all_pred.flatten()
    all_true_exp = 10 ** all_true.flatten()
    
   
    erro_percentual = np.abs((all_true_exp - all_pred_exp) / (all_true_exp + 1e-10)) * 100
    
    mape = np.mean(erro_percentual)
    mediana_erro = np.median(erro_percentual)
    std_erro = np.std(erro_percentual)
    
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

    ax1.set_xlabel("Amostra (ordenada por amplitude crescente)", fontsize=12)
    ax1.set_ylabel("Erro Percentual Absoluto (%)", fontsize=12)
    ax1.set_title(f'Erro Percentual por Amostra\n{len(all_true_exp)} amostras', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')

    # Adicionar barra de cores
    cbar = plt.colorbar(scatter, ax=ax1)
    cbar.set_label('Erro Percentual (%)', fontsize=10)

    # Adicionar segundo eixo X com a amplitude
    ax1_twin = ax1.twiny()
    ax1_twin.set_xlim(ax1.get_xlim())

    # Mostrar algumas amplitudes no eixo superior
    n_ticks = 5
    tick_positions = np.linspace(0, len(erro_sorted)-1, n_ticks, dtype=int)
    tick_labels = [f'{amp_sorted[i]:.2e}' for i in tick_positions]
    ax1_twin.set_xticks(tick_positions)
    ax1_twin.set_xticklabels(tick_labels, fontsize=8)
    ax1_twin.set_xlabel("Amplitude Real (a_gt)", fontsize=10)
    
    # PLOT 2: Índice da amostra vs Erro ======================================================
    # Ordenar por erro para ver distribuição
    indices = np.arange(len(erro_percentual))
    sorted_idx = np.argsort(erro_percentual)[::-1]  # Decrescente
    
    ax2.bar(indices, erro_percentual[sorted_idx], alpha=0.6, color='blue', edgecolor='black', linewidth=0.5)
    ax2.axhline(y=mape, color='red', linestyle='--', linewidth=2, label=f'Média: {mape:.2f}%')
    ax2.axhline(y=mediana_erro, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana_erro:.2f}%')
    
    ax2.set_xlabel("Amostra (ordenada por erro decrescente)", fontsize=12)
    ax2.set_ylabel("Erro Percentual Absoluto (%)", fontsize=12)
    ax2.set_title(f'Distribuição do Erro por Amostra\n{len(all_true_exp)} amostras', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='best')
    

    stats_text = f'MAPE: {mape:.2f}%\nMediana: {mediana_erro:.2f}%\nStd: {std_erro:.2f}%'
    ax2.text(0.95, 0.95, stats_text, 
             transform=ax2.transAxes, 
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()

    out_file = os.path.join(results_dir, "erro_percentual_por_amostra.pdf")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo em: {out_file}")
    
    # PLOT 3: Boxplot do erro ==========================================================
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot([erro_percentual], 
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    meanprops={'color': 'red', 'linestyle': '--', 'linewidth': 2},
                    medianprops={'color': 'green', 'linewidth': 2})
    
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    ax.set_xticklabels(['Erro Percentual'])
    ax.set_ylabel("Erro Percentual Absoluto (%)", fontsize=12)
    ax.set_title(f'Boxplot do Erro Percentual\n{len(all_true_exp)} amostras', fontsize=13)
    ax.grid(True, alpha=0.3, axis='y')
    

    stats_text2 = f'Média: {mape:.2f}%\nMediana: {mediana_erro:.2f}%\nStd: {std_erro:.2f}%'
    ax.text(0.95, 0.95, stats_text2, 
            transform=ax.transAxes, 
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=10)
    
    plt.tight_layout()
    
    out_file2 = os.path.join(results_dir, "boxplot_erro_percentual.pdf")
    plt.savefig(out_file2, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Boxplot salvo em: {out_file2}")
    
    # Salvar erros individuais em arquivo
    error_data = np.column_stack((
        np.arange(len(erro_percentual)),
        all_true_exp,
        all_pred_exp,
        erro_percentual
    ))
    
    header = "Amostra\tAmplitude_Real\tAmplitude_Predita\tErro_Percentual"
    np.savetxt(os.path.join(results_dir, "erros_individuals.txt"), 
               error_data, 
               fmt='%d\t%.6e\t%.6e\t%.2f',
               header=header,
               comments='')
    
    print(f"Erros individuais salvos em: {os.path.join(results_dir, 'erros_individuals.txt')}")
    
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