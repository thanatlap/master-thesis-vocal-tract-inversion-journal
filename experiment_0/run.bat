rem cd ../data_generator
rem python generating_data.py
rem cd ../experiment_0

rem python preprocess.py training ..\data\d_dataset_t3_20k di --output_path=prep_data_13 --augment_samples=0.25 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data --resample_rate=16000 --feature_normalize=4
rem python training.py 34
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_3 34_senet.h5 di --label_normalize=3

rem cd ../data_generator
rem ython csv_to_wav.py templates\predefined_param.csv di 
rem cd ../experiment_0
rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15

rem cd ../data_generator
rem python generating_data.py
rem cd ../experiment_0
rem python preprocess.py training ..\data\d_dataset_t3_40k_c di --output_path=prep_data_13 --augment_samples=0.25 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
rem python training.py 36
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data --resample_rate=16000 --feature_normalize=4
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_3 36_senet.h5 di --label_normalize=3
rem python preprocess.py predict ..\data\m_record_set mono --output_path=prep_data --resample_rate=16000 --feature_normalize=4

rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
python evaluating.py
rem python preprocess.py training ..\data\d_dataset_t3_40k_c di --output_path=prep_data_13_s --augment_samples=0.1 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
rem python training.py 37