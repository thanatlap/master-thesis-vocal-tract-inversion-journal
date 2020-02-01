python training.py 47
python preprocess.py predict ..\data\d_record_set_1 di --output_path=prep_exp10 --sample_rate=22050
python preprocess.py predict ..\data\d_record_set_2 di --output_path=prep_exp10 --sample_rate=22050
python preprocess.py predict ..\data\d_record_set_3 di --output_path=prep_exp10 --sample_rate=22050
python preprocess.py predict ..\data\d_record_set_4 di --output_path=prep_exp10 --sample_rate=22050
python preprocess.py predict ..\data\d_record_set_5 di --output_path=prep_exp10 --sample_rate=22050
python predicting.py ..\data\d_record_set_1 prep_exp10 47_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_2 prep_exp10 47_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_3 prep_exp10 47_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_4 prep_exp10 47_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_5 prep_exp10 47_cnn_bilstm.h5 di --label_normalize=3

cd ..
cd data_generator
python generating_data.py
cd ../experiment_10
python preprocess.py training ..\data\d_dataset_7 di --output_path=prep_exp10 --augment_samples=0.25 --sample_rate=22050 --label_normalize=5 --split_size=0.1
python training.py 48
python preprocess.py predict ..\data\d_record_set_1 di --output_path=prep_exp10_d7 --sample_rate=22050
python preprocess.py predict ..\data\d_record_set_2 di --output_path=prep_exp10_d7 --sample_rate=22050
python preprocess.py predict ..\data\d_record_set_3 di --output_path=prep_exp10_d7 --sample_rate=22050
python preprocess.py predict ..\data\d_record_set_4 di --output_path=prep_exp10_d7 --sample_rate=22050
python preprocess.py predict ..\data\d_record_set_5 di --output_path=prep_exp10_d7 --sample_rate=22050
python predicting.py ..\data\d_record_set_1 prep_exp10_d7 48_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_2 prep_exp10_d7 48_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_3 prep_exp10_d7 48_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_4 prep_exp10_d7 48_cnn_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_record_set_5 prep_exp10_d7 48_cnn_bilstm.h5 di --label_normalize=3



