# fno-DE3

alguns graficos gerados sobre as amplitudes
https://docs.google.com/document/d/1SsbUDqbjY68CgEQUsn-VKUKeMlNCRM00g4Vq17MCV-I/edit?tab=t.0 

git clone 
cd fno-DE3

On Linux/macOS:
source venv/bin/activate

On Windows:
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .


###Generate_data 



    -python  /generate_snl_data.py
      gera o dataset
    
    -python /plot_amplitude_histogram.py
      gera um histograma do log10 (amplitude)

    -python /plot_gamma_s.py
      grafico de relação entre log10(amplitude) e gamma
    
    -python plot_loga_logHs_logFp.py
      grafico de relação entre log10(amplitude) e log10(Hp), log10(amplitued) e log10(Fp)
      
###scale


    - python scale/run_training_min_quad.py
          treina com uma regressão linear (minimos quadraticos) (log(Hs), log(fp), log(gamma), (s))
          desenha os graficos: "scale/results_scale_loggamma_s/amplitude_original_limitado_ateX.pdf" (10**log(true_A), 10**log(pred_A))
          graficos limitados por tamanho de amostra e numero de X menores amostras
      
    -python scale/plot_erro_percentual.py
      grafico do erro porcentual e erro percetaul absoluto por amostra ordenada
      
    -python scale/metricas.py
        MAE MSE, RMSE, MAPE do log(A) e A     

        
###fno_and_a
    - python fno_and_a/polar_plot_groundtruth_fno_and_scale.py --idx=X
      grafico por amostra X, juntando fno e escala pre-treinada
