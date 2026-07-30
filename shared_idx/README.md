
## Estrutura
```texto
├── shared_idx/ 
│ ├── generate_shared_idx.py                       # gera os indices compartilhados
│ ├── results/
│ └── README.md
└──
```
   
## shared_idx
comandos de execução no diretorio shared_idx:

```texto
  python shared_idx/generate_shared_idx
  ```
  Gera a separação de amostras em um conjunto de treino e teste baseado nos indices das amotras, necessario para que o modelo de previsão de amplitue e shape tenham o mesmo conjunto de teste e validação. Ou seja, caso queira modificar o tamanho do conjunto de treino, é necessario retornar a essa parte

