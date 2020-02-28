rem cd ../data_generator
rem python generating_data.py
rem cd ../experiment_0

rem python preprocess.py training ..\data\d_dataset_t3 di --output_path=aug_data --augment_samples=0.25 --resample_rate=16000 --label_normalize=5 --feature_normalize=1 --split_size=0.15
rem python training.py 8
rem python training.py 5

rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data --resample_rate=16000 --feature_normalize=1
rem python training.py 7

rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 10_senet.h5 di --label_normalize=3
rem python training.py 10

python preprocess.py training ..\data\d_dataset_t3 di --output_path=prep_data_v2 --is_augment=False --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15