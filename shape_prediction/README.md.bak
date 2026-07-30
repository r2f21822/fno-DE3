
## Estrutura
```texto
├── shape_prediction/
│ ├── fno_diffusion/
│ ├── run_training_fno.py
│ ├── polar_plot_groundtruth_fno_and_scale.py
│ ├── results/
│ ├── figures/
│ └── README.md
└──
```
   
## shape_prediction/

### 1. Treinamento do FNO
```texto
python shape_prediction/run_training_fno.py
   ```
treina o modelo FNO normalizado

### 2. Gráficos
```texto
python shape_prediction/polar_plot_groundtruth_fno_and_scale.py
   ```
gráficos por amostra do fno e amplitude pré-treinados
parametris modificaveis:
"h5file" (default="Generate_data/snl/snl_dataset.h5")

--idx" (amostra do gafrico, default=0, help="Índice da amostra")

--model" (Caminho do modelo FNO, default="shape_prediction/results/model_best.pth")

--amplitude-model" (Caminho do modelo de amplitude, default="amlitude_prediction/linear_regression/"")

--out-dir (default="shape_prediction/figures")

--n-modes"  (default=[16, 16])

--hidden-channels(default=64)
