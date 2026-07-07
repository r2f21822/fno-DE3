import h5py
import numpy as np
import matplotlib.pyplot as plt
from fno.fno_run_training import load_snl_dataset
EPS_NORM = 1e-8

DATA_PATH = "Generate_data/snl/snl_dataset.h5"
RUN_DIR   = "fno_and_a/results_snl_factorized"

def plot_amplitude_histogram_from_h5(h5_path=DATA_PATH,out_dir=RUN_DIR,partes=50):
    """

    """

    X, Y = load_snl_dataset(h5_path)

    amplitudes = Y.abs().amax(dim=(1, 2, 3)).numpy()
    log_amplitudes = np.log10(amplitudes + 1e-10)  # log10 de a

    import os
    os.makedirs(out_dir, exist_ok=True)
 
    # Plotar histograma
    plt.figure(figsize=(8, 5))
    plt.hist(log_amplitudes, bins=partes, edgecolor='black', alpha=0.7)
   # plt.yscale('log')
    plt.xlabel("log amplitude Amplitude (max |S_nl|)")
    plt.ylabel("Contagem")
    plt.title(f"Distribuição das log_Amplitudes - {h5_path}")
    plt.grid(True, alpha=0.3)
    #plt.show()

    out_path = os.path.join(out_dir, "amplitude_histogram.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()  # Fecha a figura para liberar memória
    
    
    
    return log_amplitudes

if __name__ == "__main__":
    amps = plot_amplitude_histogram_from_h5()