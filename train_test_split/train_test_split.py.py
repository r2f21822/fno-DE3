import numpy as np
import argparse
import os
import yaml
import h5py
from sklearn.model_selection import train_test_split

DATA_PATH = "Generate_data/snl/snl_dataset.h5"
INDICES_DIR = "shared_idx/results"

def main():
    pa = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    pa.add_argument("--h5file", default=DATA_PATH)
    pa.add_argument("--out-dir", default=INDICES_DIR)
    pa.add_argument("--seed", type=int, default=42)
    pa.add_argument("--test_size", type=float, default=0.2)
    args = pa.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    
    with h5py.File(DATA_PATH, "r") as hf:
        Y = hf["Y"][:]
    n_samples = Y.shape[0]
    
    indices = np.arange(n_samples)
    
    # Dividir conjunto de treino/teste
    train_idx, val_idx = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True
    )
    
    # Salvar informações
    info = {
        'seed': args.seed,
        'test_size':args.test_size,
        'n_total': int(n_samples),
        'n_train': int(len(train_idx)),
        'n_val': int(len(val_idx)),
        'train_ratio': float(len(train_idx) / n_samples),
        'val_ratio': float(len(val_idx) / n_samples)
    }
    
    with open(os.path.join(INDICES_DIR, "indices_info.yaml"), "w") as f:
        yaml.dump(info, f, default_flow_style=False, sort_keys=False)
    
    # Salvar indices como numpy arrays
    np.save(os.path.join(INDICES_DIR, "train_indices.npy"), train_idx)
    np.save(os.path.join(INDICES_DIR, "val_indices.npy"), val_idx)
    
    print(f"Indices salvos em: {INDICES_DIR}")
    print(f"Seed: {args.seed}")
    print(f"Total: {n_samples} amostras")
    print(f"Treino: {len(train_idx)} amostras ({len(train_idx)/n_samples*100:.1f}%)")
    print(f"Validação: {len(val_idx)} amostras ({len(val_idx)/n_samples*100:.1f}%)")
 

if __name__ == "__main__":
    main()