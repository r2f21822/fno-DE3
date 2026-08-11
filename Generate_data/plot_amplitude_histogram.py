import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch


DATA_PATH = "Generate_data/snl/snl_dataset.h5"
RUN_DIR   = "Generate_data/figures"


def load_snl_dataset(path):
    with h5py.File(path, "r") as hf:
        X = hf["X"][:]   # (N, Nf, Ntheta, 1)
        Y = hf["Y"][:]   # (N, Nf, Ntheta, 1)

    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 1, Nf, Ntheta)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)  # (N, 1, Nf, Ntheta)
    return X, Y


def plot_amplitude_histogram_from_h5(h5_path=DATA_PATH,out_dir=RUN_DIR,partes=50):
   

    X, Y = load_snl_dataset(h5_path)

    amplitudes = Y.abs().amax(dim=(1, 2, 3)).numpy()
    log_amplitudes = np.log10(amplitudes)  # log10 de a

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

    save_path = os.path.join(RUN_DIR, "histograma_amplitudes_preditas_log.pdf")
    print ("histograma_amplitudes_preditas_log.pdf salvo em ",RUN_DIR )
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


    # Plotar histograma
    plt.figure(figsize=(10, 6))
    plt.hist((amplitudes), bins=100, alpha=0.7, color='red', edgecolor='black')
    plt.xlabel("Amplitude Amplitude (max |S_nl|)")
    plt.ylabel("Contagem")
    plt.title(f"Distribuição das Amplitudes - {h5_path}")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    #plt.show()

    save_path = os.path.join(RUN_DIR, "histograma_amplitudes_preditas_real.pdf")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print ("histograma_amplitudes_preditas_real.pdf salvo em ",RUN_DIR )
    return log_amplitudes

if __name__ == "__main__":
    amps = plot_amplitude_histogram_from_h5()