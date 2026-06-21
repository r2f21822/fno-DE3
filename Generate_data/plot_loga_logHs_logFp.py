import h5py
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from scale.run_training_scale import factorize_target
import os

EPS = 1e-8

DATA_PATH = "Generate_data/snl/snl_dataset.h5"
RUN_DIR   = "Generate_data"


def load_snl_dataset(path):
    with h5py.File(path, "r") as hf:
        Hs = hf["Hs"][:]
        Fp =hf["fp"][:]
        Y = hf["Y"][:]
    Hs = torch.tensor(Hs, dtype=torch.float32)
    Fp = torch.tensor(Fp, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)
    return Hs,Fp, Y

def main():
 
    Hs,Fp,Y=load_snl_dataset(DATA_PATH)

    loga = factorize_target(Y)
    loga = torch.log(loga + EPS)  

    logHs = torch.log(Hs + EPS)  
    logFp = torch.log(Fp + EPS)  

    plt.figure(figsize=(6, 5))
    plt.scatter(logHs.numpy(), loga.numpy(), alpha=0.3, s=5)
    plt.xlabel('log(Hs)')
    plt.ylabel('log(a)')
    plt.title(f'log(a) vs log(Hs) - {len(loga)} amostras')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "log_a_vs_log_Hs.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"salvo em: {os.path.join(RUN_DIR, 'log_a_vs_log_Hs.pdf')}")



    plt.figure(figsize=(6, 5))
    plt.scatter(logFp.numpy(), loga.numpy(), alpha=0.3, s=5)
    plt.xlabel('log(Fp)')
    plt.ylabel('log(a)')
    plt.title(f'log(a) vs log(fp) - {len(loga)} amostras')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "log_a_vs_log_fp.pdf"), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"salvo em: {os.path.join(RUN_DIR, 'log_a_vs_log_fp.pdf')}")




if __name__ == "__main__":
    main()