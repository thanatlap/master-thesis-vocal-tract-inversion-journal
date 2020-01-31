python predicting.py ..\data\d_record_set_1 prep_exp10 39_cnn_bilstm_20200131_0846.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_2 prep_exp10 39_cnn_bilstm_20200131_0846.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_3 prep_exp10 39_cnn_bilstm_20200131_0846.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_4 prep_exp10 39_cnn_bilstm_20200131_0846.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_5 prep_exp10 39_cnn_bilstm_20200131_0846.h5 di --label_normalize=3
python training.py 43

rem python preprocess.py predict ..\data\d_record_set_1 di --output_path=prep_exp10 --sample_rate=22050
rem python preprocess.py predict ..\data\d_record_set_2 di --output_path=prep_exp10 --sample_rate=22050
rem python preprocess.py predict ..\data\d_record_set_3 di --output_path=prep_exp10 --sample_rate=22050
rem python preprocess.py predict ..\data\d_record_set_4 di --output_path=prep_exp10 --sample_rate=22050
rem python preprocess.py predict ..\data\d_record_set_5 di --output_path=prep_exp10 --sample_rate=22050





