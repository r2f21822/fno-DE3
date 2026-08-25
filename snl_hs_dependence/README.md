
## Estrutura
```texto
├── Generate_data/
│ ├── run_linear_regression.py
│ ├── plot_and_table_hs.py
│ ├── dependencia_da_amplitude_snl_Hs.pdf
│ ├── results/
│ ├── figures/
│ └── README.md
└── 
```
   
## Geração e analise do datsep
comandos de execução no diretorio fno-DE3:

### 1. Geração do dataset
```texto
  python Generate_data/generate_data_hs_variable.py --n-samples=5000
   ```
Geração do dataset com parametros Tp, fp, gamma, th0, s fixos e Hs variando de 1m a 16m

--n-samples (Número total de amostras, default=1000)

--n-f	(Resolução de frequência (recomendada potência de 2), default=128)

--n-theta	(Resolução direcional, default=64)

--f-min	(Frequência mínima (Hz), default= 0,04)

--f-max	(Frequência máxima (Hz), default=1.0)

--seed	(Semente aleatória, default=42)

--out	(Caminho de saída, default="data/snl/snl_dataset.h5")


--Tp   (Valor de Tp fixo, default=10.0)

--gamma    (Valor de gamma fixo, default=3.3)

--th0      (Valor de th0 fixo, default=0.0)

--s        (Valor de s fixo,, default=4)

### 2. Gerar separação de treino e teste
```texto
  python train_test_split/train_test_split.py --h5file=Generate_data/snl/snl_dataset_hs_variable.h5 --out=train_test_split/results_hs
 ```

### 3. Treinar modelo
```texto
   python snl_hs_dependence/run_linear_regression.py --h5file=Generate_data/snl/snl_dataset_hs_variable.h5 --idx_dir=train_test_split/results_hs
 ```
gera também os graficos de log(a_pred) vs log(a_true) e a_pred vs a_true original

 ## 4. Geração de graficos e tabelas
 ```texto
   python snl_hs_dependence/plot_and_table_hs.py
 ```
 gera graficos de log(a) vs log(Hs), a/Hs^6, e uma tabela com os valores de Hs, log(Hs), a, log(a) e a/Hs^6