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
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 40_senet.h5 di --label_normalize=3
rem python training.py 41
rem python evaluating.py
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data --resample_rate=16000 --feature_normalize=4
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 41_senet.h5 di --label_normalize=3
rem python preprocess.py predict ..\data\m_record_set mono --output_path=prep_data --resample_rate=16000 --feature_normalize=4

rem python preprocess.py training ..\data\d_dataset_t3_40k_c di --output_path=prep_data_13_noaug --is_augment=False --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
rem python training.py 78
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data --resample_rate=16000 --feature_normalize=4
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 78_senet.h5 di --label_normalize=3
rem "result\predict_77\formant\formant_chart_disyllable FirstSyllable.png"
rem python training.py 79
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 79_senet.h5 di --label_normalize=3
rem python training.py 80
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 80_senet.h5 di --label_normalize=3
rem python training.py 81
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 81_senet.h5 di --label_normalize=3
rem python training.py 82
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 82_senet.h5 di --label_normalize=3
rem python training.py 83
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 83_senet.h5 di --label_normalize=3
rem python training.py 84
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 84_senet.h5 di --label_normalize=3
python training.py 85
python predicting.py ..\data\d_records\d_record_set_2 prep_data 85_senet.h5 di --label_normalize=3