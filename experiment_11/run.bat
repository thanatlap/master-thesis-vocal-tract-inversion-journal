rem python training.py 1
rem python training.py 3
rem python training.py 2

rem python preprocess.py predict ..\data\d_record_set_1 di --output_path=prep_exp11 --resample_rate=22050
rem python preprocess.py predict ..\data\d_record_set_2 di --output_path=prep_exp11 --resample_rate=22050
rem python preprocess.py predict ..\data\d_record_set_3 di --output_path=prep_exp11 --resample_rate=22050
rem python preprocess.py predict ..\data\d_record_set_4 di --output_path=prep_exp11 --resample_rate=22050
rem python preprocess.py predict ..\data\d_record_set_5 di --output_path=prep_exp11 --resample_rate=22050

python predicting.py ..\data\d_record_set_1 prep_exp11 2_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_2 prep_exp11 2_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_3 prep_exp11 2_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_4 prep_exp11 2_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_5 prep_exp11 2_cnn_bilstm.h5 di --label_normalize=3



