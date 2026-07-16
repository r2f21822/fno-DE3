# fno-DE3


Estrutura do projeto:
```texto
├── shape_prediction/
│ ├── fno_diffusion/
│ ├── run_training_fno.py
│ ├── polar_plot_groundtruth_fno_and_scale.py
│ ├── results/
│ ├── figures/
│ └── README.md
│
├── amplitude_prediction/
│ ├── linear_regression/
│ │ ├── run_linear_regression.py
│ │ ├── metricas.py
│ │ ├── plot_erro_por_amostra.py
│ │ ├── plot_erro_percentual.py
│ │ ├── plot_linear_regression_zoomed.py
│ │ ├── figures/
│ │ ├── results/
│ │ └── README.md
│ │
│ └── mlp/
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

## Clonar repositorio
### 1. Clonar
```texto
  git clone https://github.com/r2f21822/fno-DE3.git
  ```

### 2. Ambiente virtual
```texto
  python -m venv venv
  ```
  Linux
  ```texto
  source venv/bin/activate
  ```
  Windows
  ```texto
  venv\Scripts\activate
  ```
### 3. Instalar depencias 
  ```texto
  pip install --upgrade pip
  pip install -r requirements.txt
  ```
### 4. Instalar pacotes 
  ```texto
  pip install -e .
  ```


## Para treinar o modelo de previsão de amplitude

### 1. Gerar os dados 
```texto
  python Generate_data/generate_snl_data.py --n-sample 5000
  ```
  Snl calculado utilizando o modelo DE3. Dataset sintético baseado em espectros JONSWAP bidimensionais, sob condições oceânicas realistas de altura significativa ​entre 1 e 6 metros e período de pico entre 5 e 15 segundos.

### 2. Regressão linear de log(A)
```texto
  python amplitude_prediction/linear_regression/run_linear_regression.py
  ```
  Treina um modelo de regressão linear para prever a amplitude das amostras separadamente. Preve primeiramente o log10(amplitude), mas gera os grafico também na forma normal

### 3. Metricas e graficos de erro
```texto
  python amplitude_prediction/linear_regression/plot_erro_percentual.py
  ```
  graficos dos erros percentuais de cada amostra, ordenadas por amplitude
  ```texto
  python amplitude_prediction/linear_regression/metricas.py
  ```
   métricas de erro (MAE, MSE, RMSE, MAPE) para todas as amostras e separadamente para amplitudes abaixo e acima da mediana, além das médias e medianas dos valores reais e preditos


### 4. Zoom nos gráficos de amplitude em escala original
```texto
python amplitude_prediction/linear_regression/plot_linear_regression_zoomed.py
   ```
gera graficos da amplitude original vs a predita (em escala normal) limitados até determinados tamanhos de amplitide e quantidade de amostras

