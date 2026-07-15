-/Amplitude_predictor
	/linear_regression
        comandos de execução no diretorio fno-DE3:

		- python amplitude_prediction/linear_regression/run_linear_regression.py
            parametris modificaveis:
                        --h5file (diretorio dos dados, defult="Generate_data/snl/snl_dataset.h5")
                        --out-dir (diretorio de saida dos resultados do modelo, defult="amplitude_prediction/linear_regression/results")
                        --out-dir_figs (diretorio das figuras, defult="amplitude_prediction/linear_regression/figures/out")
                        --seed

        - python amplitude_prediction/linear_regression/plot_erro_por_amostra.py
           parametris modificaveis:
                        --dir_out (diretório de saida com as figuras, default="amplitude_prediction/linear_regression/figures/out")
                        --dir_modelo (Diretório de entrada com o modelo, default="amplitude_prediction/linear_regression/results")

        - python amplitude_prediction/linear_regression/plot_erro_percentual.py
           parametros modificaveis:
                        --dir_out (diretório de saida com as figuras, default="amplitude_prediction/linear_regression/figures/out")
                        --dir_modelo (Diretório de entrada com o modelo, default="amplitude_prediction/linear_regression/results")

        - python amplitude_prediction/linear_regression/metricas.py
           parametris modificaveis:
                        --dir_modelo (Diretório de entrada com o modelo, default="amplitude_prediction/linear_regression/results")
