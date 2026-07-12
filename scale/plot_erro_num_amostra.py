import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import h5py

MODELO_DIR = "scale/results_scale_loggamma_s"
RUN_DIR = "scale/results_scale_loggamma_s"
DATASET_FILE = "Generate_data/snl_data/snl_dataset.h5"

def metricas(results_dir, modelo_dir, dataset_file=None):
   
    all_pred = np.load(os.path.join(modelo_dir, "all_pred.npy"))
    all_true = np.load(os.path.join(modelo_dir, "all_true.npy"))
    
    all_pred_exp = 10 ** all_pred.flatten()
    all_true_exp = 10 ** all_true.flatten()
    
    # Erro percentual com sinal (positivo = subestimou, negativo = superestimou)
    erro_percentual = ((all_true_exp - all_pred_exp) / (all_true_exp + 1e-10)) * 100
    
    # Erro absoluto para estatísticas
    erro_absoluto = np.abs(erro_percentual)
    
    mape = np.mean(erro_absoluto)
    mediana_erro = np.median(erro_absoluto)
    std_erro = np.std(erro_absoluto)
    
    # ==================== CARREGAR Hs DO DATASET ====================
    Hs_values = None
    if dataset_file and os.path.exists(dataset_file):
        try:
            with h5py.File(dataset_file, "r") as hf:
                Hs_values = hf["Hs"][:]
                print(f" Hs carregado do dataset: {len(Hs_values)} amostras")
                print(f"   Hs - min: {Hs_values.min():.2f}, max: {Hs_values.max():.2f}, média: {Hs_values.mean():.2f}")
        except Exception as e:
            print(f" Erro ao carregar Hs: {e}")
            Hs_values = None
    else:
        print(f"Dataset não encontrado: {dataset_file}")
        print("   Plot ordenado por Hs não será gerado")
    
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # ==================== PLOT 1: Erro por AMOSTRA ORIGINAL ====================
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
    ax1.set_title(f'Erro por Amostra Original\n(verde=subestimou, vermelho=superestimou)', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best', fontsize=9)
    
    stats_text = f'MAPE: {mape:.2f}%\nMediana: {mediana_erro:.2f}%\nStd: {std_erro:.2f}%'
    ax1.text(0.02, 0.98, stats_text, 
             transform=ax1.transAxes, 
             verticalalignment='top', horizontalalignment='left',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
             fontsize=9)
    
    # ==================== PLOT 2: Erro ordenado por Hs ====================
    if Hs_values is not None:
        if len(Hs_values) != len(erro_percentual):
            print(f" Tamanho do Hs ({len(Hs_values)}) não bate com erro ({len(erro_percentual)})")
            
            min_len = min(len(Hs_values), len(erro_percentual))
            Hs_values = Hs_values[:min_len]
            erro_percentual_hs = erro_percentual[:min_len]
            erro_absoluto_hs = erro_absoluto[:min_len]
        else:
            erro_percentual_hs = erro_percentual
            erro_absoluto_hs = erro_absoluto
        
  
        sorted_idx_hs = np.argsort(Hs_values)
        Hs_sorted = Hs_values[sorted_idx_hs]
        erro_sorted_hs = erro_percentual_hs[sorted_idx_hs]
        
   
        cores_hs = ['green' if e > 0 else 'red' for e in erro_sorted_hs]
        
   
        ax2.scatter(Hs_sorted, erro_sorted_hs, 
                    alpha=0.6, s=25, c=cores_hs, edgecolors='black', linewidth=0.5)
        
     
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.7, label='Erro zero')
        ax2.axhline(y=mape, color='red', linestyle='--', linewidth=2, 
                    label=f'MAPE: {mape:.2f}%')
        ax2.axhline(y=-mape, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
        ax2.axhline(y=mediana_erro, color='green', linestyle='--', linewidth=2, 
                    label=f'Mediana: {mediana_erro:.2f}%')
        ax2.axhline(y=-mediana_erro, color='green', linestyle='--', linewidth=1.5, alpha=0.5)
     
        z = np.polyfit(Hs_sorted, erro_sorted_hs, 1)
        p = np.poly1d(z)
        ax2.plot(Hs_sorted, p(Hs_sorted), 'orange', linestyle='-', alpha=0.7, linewidth=2,
                 label=f'Tendência: slope={z[0]:.2f}')
        
        ax2.set_xlabel("Hs (Altura Significativa da Onda) [m]", fontsize=12)
        ax2.set_ylabel("Erro Percentual (%)", fontsize=12)
        ax2.set_title(f'Erro vs Hs\n(verde=subestimou, vermelho=superestimou)', fontsize=13)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='best', fontsize=9)
        
        # Estatísticas no gráfico
        ax2.text(0.02, 0.98, stats_text, 
                 transform=ax2.transAxes, 
                 verticalalignment='top', horizontalalignment='left',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=9)
        
    
        correlacao = np.corrcoef(Hs_sorted, erro_absoluto_hs[sorted_idx_hs])[0, 1]
        ax2.text(0.98, 0.02, f'Correlação |erro| vs Hs: {correlacao:.3f}', 
                 transform=ax2.transAxes, 
                 verticalalignment='bottom', horizontalalignment='right',
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                 fontsize=9)
        
     
        print("\n" + "="*60)
        print("ANÁLISE DO ERRO POR FAIXA DE Hs")
        print("="*60)
        faixas_hs = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
        for hs_min, hs_max in faixas_hs:
            mask = (Hs_values >= hs_min) & (Hs_values < hs_max)
            if np.sum(mask) > 0:
                erro_medio = np.mean(erro_absoluto[mask])
                erro_std = np.std(erro_absoluto[mask])
                n_amostras = np.sum(mask)
                erro_medio_sinal = np.mean(erro_percentual[mask])
                tendencia = "subestima" if erro_medio_sinal > 0 else "superestima"
                print(f"Hs [{hs_min:.0f}, {hs_max:.0f}): n={n_amostras:3d}, "
                      f"MAPE={erro_medio:.2f}% ± {erro_std:.2f}%, "
                      f"tendência: {tendencia} ({erro_medio_sinal:+.2f}%)")
    else:
        
        ax2.text(0.5, 0.5, 'Hs não disponível\nPara gerar este plot,\nforneça o dataset com Hs',
                 horizontalalignment='center', verticalalignment='center',
                 transform=ax2.transAxes, fontsize=14, color='gray')
        ax2.set_title('Erro vs Hs (dados indisponíveis)', fontsize=13)
        ax2.set_xlabel("Hs (m)", fontsize=12)
        ax2.set_ylabel("Erro Percentual (%)", fontsize=12)
    
    plt.tight_layout()
    
    
    out_file = os.path.join(results_dir, "erro_por_amostra_com_hs.pdf")
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
    

    if Hs_values is not None and len(Hs_values) == len(erro_percentual):
        error_data = np.column_stack((
            np.arange(len(erro_percentual)),  
            Hs_values,
            all_true_exp,
            all_pred_exp,
            erro_percentual,
            erro_absoluto
        ))
        header = "Amostra\tHs\tAmplitude_Real\tAmplitude_Predita\tErro_Percentual\tErro_Absoluto"
    else:
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
               fmt='%d\t' + '\t'.join(['%.6e']*(error_data.shape[1]-1)),
               header=header,
               comments='')
    
    print(f"Erros individuais salvos em: {os.path.join(results_dir, 'erros_individuals.txt')}")
    
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
    parser.add_argument("--dataset", type=str, help="Caminho para o dataset HDF5 (para carregar Hs)", 
                        default=DATASET_FILE)
    args = parser.parse_args()
    
    metricas(args.dir_out, args.dir_modelo, args.dataset)