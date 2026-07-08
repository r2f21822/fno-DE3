import numpy as np
import matplotlib.pyplot as plt
import os

def plot_histograma_preditas(results_dir):
    
    all_pred = np.load(os.path.join(results_dir, "all_pred.npy"))
    
  
    all_pred_exp = 10 ** all_pred
    

    plt.figure(figsize=(10, 6))
    plt.hist(all_pred_exp, bins=100, alpha=0.7, color='blue', edgecolor='black')
    
    plt.xlabel('Amplitude Predita')
    plt.ylabel('Frequência')
    plt.title(f'Histograma das Amplitudes Preditas\n{len(all_pred_exp)} amostras')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = os.path.join(results_dir, "histograma_amplitudes_preditas.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Histograma salvo em: {save_path}")

    
    return all_pred_exp

if __name__ == "__main__":
    RESULTS_DIR = "scale/results_scale_loggamma_s"
    plot_histograma_preditas(RESULTS_DIR)