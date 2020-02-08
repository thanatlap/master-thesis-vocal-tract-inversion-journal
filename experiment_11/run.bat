cd ..
cd data_generator
python generating_data.py
cd ../experiment_11

python preprocess.py training ..\data\d_dataset_12 di --output_path=prep_exp11 --augment_samples=0.3 --resample_rate=22050 --label_normalize=5 --feature_normalize=3 --split_size=0.1

python training.py 1
python training.py 5
python training.py 2
python training.py 7

python preprocess.py predict ..\data\d_records\d_record_set_1 di --output_path=prep_exp11 --resample_rate=22050 --feature_normalize=3
python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_exp11 --resample_rate=22050 --feature_normalize=3
python preprocess.py predict ..\data\d_records\d_record_set_3 di --output_path=prep_exp11 --resample_rate=22050 --feature_normalize=3
python preprocess.py predict ..\data\d_records\d_record_set_4 di --output_path=prep_exp11 --resample_rate=22050 --feature_normalize=3
python preprocess.py predict ..\data\d_records\d_record_set_5 di --output_path=prep_exp11 --resample_rate=22050 --feature_normalize=3
python preprocess.py predict ..\data\d_records\d_record_set_6 di --output_path=prep_exp11 --resample_rate=22050 --feature_normalize=3
python preprocess.py predict ..\data\d_records\d_record_set_7 di --output_path=prep_exp11 --resample_rate=22050 --feature_normalize=3
python preprocess.py predict ..\data\d_records\d_record_set_8 di --output_path=prep_exp11 --resample_rate=22050 --feature_normalize=3

python predicting.py ..\data\d_records\d_record_set_1 prep_exp11 21_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_exp11 21_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_exp11 21_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_exp11 21_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_exp11 21_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_exp11 21_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_exp11 21_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_exp11 21_bilstm.h5 di --label_normalize=3

python predicting.py ..\data\d_records\d_record_set_1 prep_exp11 25_bilstm_2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_exp11 25_bilstm_2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_exp11 25_bilstm_2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_exp11 25_bilstm_2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_exp11 25_bilstm_2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_exp11 25_bilstm_2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_exp11 25_bilstm_2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_exp11 25_bilstm_2.h5 di --label_normalize=3

python predicting.py ..\data\d_records\d_record_set_1 prep_exp11 22_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_exp11 22_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_exp11 22_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_exp11 22_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_exp11 22_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_exp11 22_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_exp11 22_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_exp11 22_cnn_bilstm.h5 di --label_normalize=3

python predicting.py ..\data\d_records\d_record_set_1 prep_exp11 27_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_exp11 27_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_exp11 27_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_exp11 27_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_exp11 27_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_exp11 27_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_exp11 27_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_exp11 27_cnn_bilstm.h5 di --label_normalize=3

python training.py 3
python training.py 4
python training.py 6
python training.py 8

