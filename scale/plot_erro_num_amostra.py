import numpy as np
import matplotlib.pyplot as plt
import os
import argparse

MODELO_DIR = "scale/results_scale_loggamma_s"
RUN_DIR = "scale/results_scale_loggamma_s"

def metricas(results_dir, modelo_dir):
   
    all_pred = np.load(os.path.join(modelo_dir, "all_pred.npy"))
    all_true = np.load(os.path.join(modelo_dir, "all_true.npy"))
    
    all_pred_exp = 10 ** all_pred.flatten()
    all_true_exp = 10 ** all_true.flatten()
    

    erro_percentual = ((all_true_exp - all_pred_exp) / (all_true_exp + 1e-10)) * 100
    

    erro_absoluto = np.abs(erro_percentual)
    
    mape = np.mean(erro_absoluto)
    mediana_erro = np.median(erro_absoluto)
    std_erro = np.std(erro_absoluto)
    
    print("="*60)
    print("ESTATÍSTICAS DO ERRO PERCENTUAL")
    print("="*60)
    print(f"MAPE (Média absoluta): {mape:.2f}%")
    print(f"Mediana:               {mediana_erro:.2f}%")
    print(f"Desvio padrão:         {std_erro:.2f}%")
    print(f"Mínimo:                {erro_percentual.min():.2f}%")
    print(f"Máximo:                {erro_percentual.max():.2f}%")
    print(f"Q1 (25%):              {np.percentile(erro_absoluto, 25):.2f}%")
    print(f"Q3 (75%):              {np.percentile(erro_absoluto, 75):.2f}%")
    print(f"Superestimou (>0):     {np.sum(erro_percentual > 0)} amostras ({np.sum(erro_percentual > 0)/len(erro_percentual)*100:.1f}%)")
    print(f"Subestimou (<0):       {np.sum(erro_percentual < 0)} amostras ({np.sum(erro_percentual < 0)/len(erro_percentual)*100:.1f}%)")
    print("="*60)
    
    # ==================== FIGURA COM 2 PLOTS ====================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # ==================== PLOT 1: Erro por AMOSTRA ORIGINAL ====================
    # Eixo X = índice original da amostra (0 a 999)
    indices_originais = np.arange(len(erro_percentual))
    
    
    cores = ['green' if e > 0 else 'red' for e in erro_percentual]

    ax1.scatter(indices_originais, erro_percentual, 
                alpha=0.6, s=25, c=cores, edgecolors='black', linewidth=0.5)
    
   
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='Erro zero')
    

    ax1.axhline(y=mape, color='red', linestyle='--', linewidth=2, 
                label=f'MAPE: {mape:.2f}%')
    ax1.axhline(y=-mape, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax1.axhline(y=mediana_erro, color='green', linestyle='--', linewidth=2, 
                label=f'Mediana: {mediana_erro:.2f}%')
    ax1.axhline(y=-mediana_erro, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    

    ax1.set_xlabel("Índice da Amostra (original do dataset)", fontsize=12)
    ax1.set_ylabel("Erro Percentual (%)", fontsize=12)
    ax1.set_title(f'Erro por Amostra (verde=subestimou, vermelho=superestimou)\n{len(erro_percentual)} amostras', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    
    stats_text = f'MAPE: {mape:.2f}%\nMediana: {mediana_erro:.2f}%\nStd: {std_erro:.2f}%'
    ax1.text(0.02, 0.98, stats_text, 
             transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    # ==================== PLOT 2: Distribuição do Erro (ordenado) ====================

    sorted_idx = np.argsort(erro_absoluto)[::-1]
    erro_sorted = erro_percentual[sorted_idx]
    

    cores_barras = ['green' if e > 0 else 'red' for e in erro_sorted]
    
    ax2.bar(np.arange(len(erro_sorted)), erro_sorted, 
            alpha=0.6, color=cores_barras, edgecolor='black', linewidth=0.5)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=mape, color='red', linestyle='--', linewidth=2, 
                label=f'MAPE: {mape:.2f}%')
    ax2.axhline(y=-mape, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
    ax2.axhline(y=mediana_erro, color='green', linestyle='--', linewidth=2, 
                label=f'Mediana: {mediana_erro:.2f}%')
    ax2.axhline(y=-mediana_erro, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
    
    ax2.set_xlabel("Amostra (ordenada por erro decrescente)", fontsize=12)
    ax2.set_ylabel("Erro Percentual (%)", fontsize=12)
    ax2.set_title(f'Distribuição do Erro (ordenado)\n{len(erro_percentual)} amostras', fontsize=13)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend(loc='best')
    
  
    ax2.text(0.02, 0.98, stats_text, 
             transform=ax2.transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=10)
    
    plt.tight_layout()
    
  
    out_file = os.path.join(results_dir, "erro_por_amostra_original.pdf")
    plt.savefig(out_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f" Gráfico salvo em: {out_file}")
    
    # ==================== PLOT 3: Boxplot ====================
    fig2, ax = plt.subplots(figsize=(8, 6))
    
    bp = ax.boxplot([erro_percentual], 
                    patch_artist=True,
                    showmeans=True,
                    meanline=True,
                    meanprops={'color': 'red', 'linestyle': '--', 'linewidth': 2},
                    medianprops={'color': 'green', 'linewidth': 2})
    
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][0].set_alpha(0.7)
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax.set_xticklabels(['Erro Percentual'])
    ax.set_ylabel("Erro Percentual (%)", fontsize=12)
    ax.set_title(f'Boxplot do Erro Percentual\n{len(erro_percentual)} amostras', fontsize=13)
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
    print(f" Boxplot salvo em: {out_file2}")
    

    error_data = np.column_stack((
        np.arange(len(erro_percentual)), 
        all_true_exp,
        all_pred_exp,
        erro_percentual,
        erro_absoluto
    ))
    
    header = "Amostra\tAmplitude_Real\tAmplitude_Predita\tErro_Percentual\tErro_Absoluto"
    np.savetxt(os.path.join(results_dir, "erros_individuals.txt"), 
               error_data, 
               fmt='%d\t%.6e\t%.6e\t%.2f\t%.2f',
               header=header,
               comments='')
    
    print(f" Erros individuais salvos em: {os.path.join(results_dir, 'erros_individuals.txt')}")
    
    return {
        'mape': mape,
        'mediana': mediana_erro,
        'std': std_erro,
        'min': erro_percentual.min(),
        'max': erro_percentual.max(),
        'q1': np.percentile(erro_absoluto, 25),
        'q3': np.percentile(erro_absoluto, 75)
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir_out", type=str, help="Diretório para salvar os resultados", default=RUN_DIR)
    parser.add_argument("--dir_modelo", type=str, help="Diretório com as métricas", default=MODELO_DIR)
    args = parser.parse_args()
    
    metricas(args.dir_out, args.dir_modelo)