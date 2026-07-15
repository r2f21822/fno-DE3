# fno-DE3


Estrutura do projeto:
```texto
├── shape_prediction/
│ ├── fno_and_a/
│ │ ├──
│ │ └── README.md
│ └── fno_diffusion
│ ├──
│ └── README.md
│
├── amplitude_prediction/
│ ├── linear_regression/
│ │ ├── run_linear_regression.py
│ │ ├── metricas.py
│ │ ├── plot_erro_por_amostra.py
│ │ ├── plot_erro_percentual.py
│ │ ├── figures/
│ │ ├── results/
│ │ └── README.md
│ │
│ └── mlp/
│
│
├── Generate_data/
│ ├── generate_snl_data.py
│ ├── plt_amplitude_histograma.py
│ ├── plot_gamma.s.py
│ ├── plot_loga_logHs_logFp
│ ├── snl_physics.py
│ ├── snl/
│ ├── results/
│ └── README.md
│
└── README.md
```

# Para treinar o modelo

1. Gerar os dados 
```texto
  python Generate_data/generate_snl_data.py --n-sample 5000
  ```
2. Regressão linear de log(A)
```texto
  python amplitude_prediction/linear_regression/run_linear_regression.py
  ```