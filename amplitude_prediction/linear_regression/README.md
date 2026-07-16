
## Estrutura
```texto
├── amplitude_prediction/
│ ├── linear_regression/
│ │ ├── run_linear_regression.py
│ │ ├── plot_erro_por_amostra.py
│ │ ├── plot_erro_percentual.py
│ │ ├── plot_linear_regression_zoomed.py
│ │ ├── metricas.py
│ │ ├── figures/
│ │ ├── results/
│ │ └── README.md
│ │
│ └── mlp/
└──
```
   
## linear_regression
comandos de execução no diretorio fno-DE3:

### 1. Treino
```texto
python amplitude_prediction/linear_regression/run_linear_regression.py
   ```
treina um modelo de regressão linear para prever o log10 da amplitude, tambem gera dos graficos, um de log10(amplitude_predita) vs log10(amplitude real) e um amplitude_predita vs amplitude_real
parametros modificaveis:
--h5file (diretorio dos dados, defult="Generate_data/snl/snl_dataset.h5")
--out-dir (diretorio de saida dos resultados do modelo, defult="amplitude_prediction/linear_regression/results")
--out-dir_figs (diretorio das figuras, defult="amplitude_prediction/linear_regression/figures/out")
--seed

### 2. Gráficos           
```texto
python amplitude_prediction/linear_regression/plot_erro_por_amostra.py
   ```
parametros modificaveis:
--dir_out (diretório de saida com as figuras, default="amplitude_prediction/linear_regression/figures")


--dir_modelo (Diretório de entrada com o modelo, default="amplitude_prediction/linear_regression/results")

```texto
python amplitude_prediction/linear_regression/plot_erro_percentual.py
   ```
parametros modificaveis:
--dir_out (diretório de saida com as figuras, default="amplitude_prediction/linear_regression/figures")



--dir_modelo (Diretório de entrada com o modelo, default="amplitude_prediction/linear_regression/results")


```texto
python amplitude_prediction/linear_regression/plot_linear_regression_zoomed.py
   ```
gera graficos da amplitude original vs a predita (em escala normal) limitados até determinados tamanhos de amplitide e quantidade de amostras
parametros modificaveis:
--out_dir_figs (diretório de saida com as figuras, default="amplitude_prediction/linear_regression/figures")



--dir_modelo (Diretório de entrada com o modelo, default="amplitude_prediction/linear_regression/results")

### 3. Metricas no terminal
```texto
python amplitude_prediction/linear_regression/metricas.py
   ```
parametris modificaveis:
 --dir_modelo (Diretório de entrada com o modelo, default="amplitude_prediction/linear_regression/results")
            