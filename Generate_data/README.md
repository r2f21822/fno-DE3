
## Estrutura
```texto
├── Generate_data/
│ ├── generate_snl_data.py
│ ├── plt_amplitude_histograma.py
│ ├── plot_gamma.s.py
│ ├── plot_loga_logHs_logFp
│ ├── snl_physics.py
│ ├── snl/
│ ├── figures/
│ └── README.md
│
└── README.md
```
   
## Geração e analise do datsep
comandos de execução no diretorio fno-DE3:

### 1. Geração
```texto
  python Generate_data/generate_snl_data.py --n-samples=5000
   ```
--n-samples (Número total de amostras, default=1000)

--n-f	(Resolução de frequência (recomendada potência de 2), default=128)

--n-theta	(Resolução direcional, default=64)

--f-min	(Frequência mínima (Hz), default= 0,04)

--f-max	(Frequência máxima (Hz), default=1.0)

--seed	(Semente aleatória, default=42)

--out	(Caminho de saída, default="data/snl/snl_dataset.h5")


### 2. Gráficos de analise          
```texto
python Generate_data/plot_amplitude_histogram.py
   ```
histograma da distribuição de amplitudes em log10 e real

```texto
python Generate_data/plot_gamma_s.py
   ```
gráficos da relação de gamma e s com log(amplitude)

```texto
python Generate_data/plot_loga_logHs_logFp.py
```
gráficos da relação de log(Hps) e log(Fp) com log(amplitude)
