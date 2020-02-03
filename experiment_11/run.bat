cd ..
cd data_generator
python generating_data.py
cd ../experiment_11

python preprocess.py training ..\data\d_dataset_9 di --output_path=prep_exp11 --augment_samples=0.20 --resample_rate=22050 --label_normalize=5 --split_size=0.1

python training.py 1
python training.py 2

python preprocess.py predict ..\data\d_record_set_1 di --output_path=prep_exp11 --resample_rate=22050
python preprocess.py predict ..\data\d_record_set_2 di --output_path=prep_exp11 --resample_rate=22050
python preprocess.py predict ..\data\d_record_set_3 di --output_path=prep_exp11 --resample_rate=22050
python preprocess.py predict ..\data\d_record_set_4 di --output_path=prep_exp11 --resample_rate=22050
python preprocess.py predict ..\data\d_record_set_5 di --output_path=prep_exp11 --resample_rate=22050

python predicting.py ..\data\d_record_set_1 prep_exp11 4_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_2 prep_exp11 4_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_3 prep_exp11 4_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_4 prep_exp11 4_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_5 prep_exp11 4_cnn_bilstm.h5 di --label_normalize=3



