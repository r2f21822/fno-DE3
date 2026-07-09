# scripts/run_training_with_amplitude.py

import os
import json
import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from fno_diffusion.model import make_fno_2d


DATA_PATH = "Generate_data/snl_data/snl_dataset.h5"
RUN_DIR = "fno_and_a/results_fno_and_a"

# Caminho do modelo de amplitude treinado
AMPLITUDE_MODEL_PATH = "scale/results_scale_loggamma_s/model_best.pth"  # ou o caminho do seu melhor modelo

CONFIG = {
    "model": {
        "type": "FNO_2D_with_amplitude",
        "n_modes": (16, 16),
        "hidden_channels": 64,
    },
    "training": {
        "epochs": 100,
        "batch_size": 16,
        "learning_rate": 1e-3,
        "freeze_amplitude": True,  # Congela o modelo de amplitude
    },
    "data": {
        "path": DATA_PATH,
    },
}

# ---------------------------------------------------------------------------
# Modelo de Amplitude (igual ao seu código)
# ---------------------------------------------------------------------------

EPS = 1e-8

class LinearRegressor(nn.Module):
    """Modelo que prevê amplitude a partir de log(Hs), log(fp), log(gamma), s"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, x):
        return self.linear(x)

def load_hs_fp_dataset(path):
    with h5py.File(path, "r") as hf:
        Hs = hf["Hs"][:]   # (N,)
        fp = hf["fp"][:]   # (N,)
        gamma = hf["gamma"][:]   # (N,)
        s = hf["s"][:]   # (N,)
        Y = hf["Y"][:]
    
    Hs = torch.tensor(Hs, dtype=torch.float32)
    fp = torch.tensor(fp, dtype=torch.float32)
    s = torch.tensor(s, dtype=torch.float32)
    gamma = torch.tensor(gamma, dtype=torch.float32)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)
    
    return Hs, fp, Y, gamma, s

def factorize_target(Y, eps=EPS):
    """Calcula a amplitude máxima de cada amostra"""
    a = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + eps
    return a.view(-1, 1)

def denormalize_with_amplitude(Y_norm, a):
    """Reconstroi S_nl = Y_norm * a"""
    return Y_norm * a.view(-1, 1, 1, 1)

# ---------------------------------------------------------------------------
# Modelo FNO com Amplitude
# ---------------------------------------------------------------------------

class FNOWithAmplitude(nn.Module):
    """FNO que aprende a forma normalizada e usa amplitude pré-treinada"""
    def __init__(self, n_modes=(16, 16), hidden_channels=64):
        super().__init__()
        # FNO para a forma normalizada
        self.field_model = make_fno_2d(
            n_modes=n_modes,
            hidden_channels=hidden_channels,
            in_channels=1,
            out_channels=1,
        )
        # Modelo de amplitude (será carregado depois)
        self.amplitude_model = LinearRegressor()
        
    def forward(self, x_fno, x_amplitude):
        """
        x_fno: entrada para o FNO (campo E)
        x_amplitude: entrada para o modelo de amplitude [log(Hs), log(fp), log(gamma), s]
        """
        # Prever a forma normalizada
        y_tilde = self.field_model(x_fno)
        
        # Prever a amplitude
        a_pred = self.amplitude_model(x_amplitude)
        a_pred = torch.exp(a_pred)  # Converter de log para escala original (se o modelo treinou em log)
        
        # Reconstruir
        y_pred = y_tilde * a_pred.view(-1, 1, 1, 1)
        return y_pred, y_tilde, a_pred

# ---------------------------------------------------------------------------
# Normalização e Losses
# ---------------------------------------------------------------------------

EPS_NORM = 1e-8

def normalize_per_sample(Y, eps=EPS_NORM):
    """Normaliza cada amostra pelo seu máximo absoluto"""
    scale = Y.abs().amax(dim=(1, 2, 3), keepdim=True) + eps
    return Y / scale, scale

def relative_l2_loss(pred, target, eps=EPS_NORM):
    """Relative L2 padrão"""
    diff = (pred - target).view(pred.size(0), -1)
    tgt = target.view(target.size(0), -1)
    return (diff.norm(dim=1) / (tgt.norm(dim=1) + eps)).mean()

def load_snl_dataset(path):
    with h5py.File(path, "r") as hf:
        X = hf["X"][:]   # (N, Nf, Ntheta, 1)
        Y = hf["Y"][:]   # (N, Nf, Ntheta, 1)

    X = torch.tensor(X, dtype=torch.float32).permute(0, 3, 1, 2)
    Y = torch.tensor(Y, dtype=torch.float32).permute(0, 3, 1, 2)
    return X, Y

def compute_stats(tensor):
    return (
        float(tensor.min()),
        float(tensor.max()),
        float(tensor.mean()),
        float(tensor.std()),
    )

def print_stats(name, min_, max_, mean_, std_):
    print(f"  {name}:")
    print(f"    min  = {min_:.4e}")
    print(f"    max  = {max_:.4e}")
    print(f"    mean = {mean_:.4e}")
    print(f"    std  = {std_:.4e}")

def save_curve(train_values, val_values, out_path, ylabel, title):
    plt.figure(figsize=(7, 4.5))
    plt.plot(train_values, label="Train")
    plt.plot(val_values, label="Validation")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on device: {device}")

    os.makedirs(RUN_DIR, exist_ok=True)

    # Salvar config
    with open(os.path.join(RUN_DIR, "config.yaml"), "w") as f:
        yaml.dump(CONFIG, f, sort_keys=False)

    # Carregar dados
    X, Y = load_snl_dataset(DATA_PATH)
    Hs, fp, Y_orig, gamma, s = load_hs_fp_dataset(DATA_PATH)
    
    print(f"Dataset shape — X: {X.shape}  Y: {Y.shape}")

    # ========== PREPARAR DADOS PARA O MODELO DE AMPLITUDE ==========
    # Calcular amplitude real (para referência)
    a_real = factorize_target(Y)  # shape (N, 1)
    a_real_log = torch.log10(a_real + EPS)  # log da amplitude
    
    # Preparar entrada para o modelo de amplitude
    X_amplitude = torch.stack([
        torch.log10(Hs + EPS),
        torch.log10(fp + EPS),
        torch.log10(gamma + EPS),
        s
    ], dim=1)  # shape (N, 4)

    # ========== CARREGAR MODELO DE AMPLITUDE PRÉ-TRENADO ==========
    amplitude_model = LinearRegressor().to(device)
    
    if os.path.exists(AMPLITUDE_MODEL_PATH):
        print(f"Carregando modelo de amplitude de: {AMPLITUDE_MODEL_PATH}")
        state_dict = torch.load(AMPLITUDE_MODEL_PATH, map_location=device)
        amplitude_model.load_state_dict(state_dict)
        amplitude_model.eval()
        
        # Congelar o modelo de amplitude
        if CONFIG["training"]["freeze_amplitude"]:
            for param in amplitude_model.parameters():
                param.requires_grad = False
            print("✅ Modelo de amplitude congelado (não será treinado)")
    else:
        print(f"⚠️ Modelo de amplitude não encontrado em: {AMPLITUDE_MODEL_PATH}")
        print("Treinando o modelo de amplitude do zero...")
        # Se não encontrar, vai treinar junto com o FNO

    # ========== NORMALIZAÇÃO ==========
    Y_norm, Y_scale = normalize_per_sample(Y)
    
    print(f"Escala |Y|.max — min={Y_scale.min():.3e}  max={Y_scale.max():.3e}  mean={Y_scale.mean():.3e}")

    # ========== DIVIDIR DADOS ==========
    n_total = len(X)
    n_train = int(0.8 * n_total)
    indices = torch.randperm(n_total, generator=torch.Generator().manual_seed(42))
    train_idx = indices[:n_train]
    val_idx = indices[n_train:]

    # Datasets: X_fno, Y_norm, Y_scale, X_amplitude, a_real
    train_ds = TensorDataset(
        X[train_idx], 
        Y_norm[train_idx], 
        Y_scale[train_idx],
        X_amplitude[train_idx],
        a_real_log[train_idx]
    )
    val_ds = TensorDataset(
        X[val_idx], 
        Y_norm[val_idx], 
        Y_scale[val_idx],
        X_amplitude[val_idx],
        a_real_log[val_idx]
    )

    train_loader = DataLoader(train_ds, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG["training"]["batch_size"])

    # ========== MODELO FNO COM AMPLITUDE ==========
    model = FNOWithAmplitude(
        n_modes=CONFIG["model"]["n_modes"],
        hidden_channels=CONFIG["model"]["hidden_channels"],
    ).to(device)

    # CARREGAR O MODELO DE AMPLITUDE NO MODELO PRINCIPAL
    model.amplitude_model.load_state_dict(amplitude_model.state_dict())
    
    if CONFIG["training"]["freeze_amplitude"]:
        for param in model.amplitude_model.parameters():
            param.requires_grad = False

    # ========== OTIMIZADOR ==========
    # Apenas parâmetros do FNO (e amplitude se não estiver congelada)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=CONFIG["training"]["learning_rate"])
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_losses, val_losses = [], []
    best_val = float("inf")

    # ========== TREINAMENTO ==========
    for epoch in range(CONFIG["training"]["epochs"]):
        model.train()
        train_loss = 0.0
        
        for xb, yb_norm, yb_scale, xb_amp, ab_log in train_loader:
            xb = xb.to(device)
            yb_norm = yb_norm.to(device)
            xb_amp = xb_amp.to(device)
            
            optimizer.zero_grad()
            
            # Forward
            y_pred, y_tilde, a_pred = model(xb, xb_amp)
            
            # Loss: apenas na forma normalizada
            loss = relative_l2_loss(y_tilde, yb_norm)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * xb.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # ========== VALIDAÇÃO ==========
        model.eval()
        val_loss = 0.0
        val_phys_loss = 0.0
        
        with torch.no_grad():
            for xb, yb_norm, yb_scale, xb_amp, ab_log in val_loader:
                xb = xb.to(device)
                yb_norm = yb_norm.to(device)
                yb_scale = yb_scale.to(device)
                xb_amp = xb_amp.to(device)
                
                y_pred, y_tilde, a_pred = model(xb, xb_amp)
                
                # Loss da forma normalizada
                loss = relative_l2_loss(y_tilde, yb_norm)
                val_loss += loss.item() * xb.size(0)
                
                # Reconstrução física para métrica
                y_phys = denormalize_with_amplitude(y_tilde, a_real[val_idx] if epoch == 0 else None)
                # (Aqui você pode calcular physical loss se quiser)
        
        val_loss /= len(val_loader.dataset)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
        
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "model_best.pth"))
        
        print(f"Epoch {epoch:03d} | Train: {train_loss:.4e} | Val: {val_loss:.4e}")
    
    # ========== SALVAR MODELO ==========
    torch.save(model.state_dict(), os.path.join(RUN_DIR, "model.pth"))
    
    # ========== SALVAR MÉTRICAS ==========
    metrics = {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "best_val_loss": best_val,
        "amplitude_model": {
            "path": AMPLITUDE_MODEL_PATH,
            "frozen": CONFIG["training"]["freeze_amplitude"],
        },
        "config": CONFIG,
    }
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # ========== CURVAS DE LOSS ==========
    save_curve(train_losses, val_losses, os.path.join(RUN_DIR, "loss_curves.pdf"), 
               "Relative L2", "FNO Loss (forma normalizada)")
    
    print(f"\n Treinamento finalizado!")
    print(f" Arquivos salvos em: {RUN_DIR}")
    print(f" Best validation loss: {best_val:.4e}")

if __name__ == "__main__":
    main()