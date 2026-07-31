
## Estrutura
```texto
├── train_test_split/ 
│ ├── train_test_split.py                       # gera os indices compartilhados
│ ├── results/
│ └── README.md
└──
```
   
## shared_idx
comandos de execução no diretorio shared_idx:

```texto
  python train_test_split/train_test_split.py
  ```
  Gera a separação de amostras em um conjunto de treino e teste baseado nos indices das amotras, necessario para que o modelo de previsão de amplitue e shape tenham o mesmo conjunto de teste e validação. Ou seja, caso queira modificar o tamanho do conjunto de treino, é necessario retornar a essa parte
   --h5file" (default=DATA_PATH)
   --out-dir" (default=INDICES_DIR)
   --seed" (default=42)
   --test_size" (default=0.2)
