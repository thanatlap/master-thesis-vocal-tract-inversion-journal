rem cd ../data_generator
rem python generating_data.py
rem cd ../experiment_0

rem python preprocess.py training ..\data\d_dataset_t3_20k di --output_path=prep_data_13 --augment_samples=0.25 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data --resample_rate=16000 --feature_normalize=4
rem python training.py 34
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_3 34_senet.h5 di --label_normalize=3

rem cd ../data_generator
rem python csv_to_wav.py templates\predefined_param.csv di 
rem cd ../experiment_0
rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15

rem cd ../data_generator
rem python generating_data.py
rem cd ../experiment_0
rem python preprocess.py training ..\data\d_dataset_t3_40k_c di --output_path=prep_data_13 --augment_samples=0.25 --resample_rate=16000 --label_normalize=5 --feature_normalize=4 --split_size=0.15
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 40_senet.h5 di --label_normalize=3
rem python training.py 41
rem python evaluating.py
python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data --mfcc_coef=20
python predicting.py ..\data\d_records\d_record_set_2 prep_data 21_senet.h5 di --label_normalize=3
rem python preprocess.py predict ..\data\m_record_set mono --output_path=prep_data --resample_rate=16000 --feature_normalize=4

rem cd generator
rem python generating.py
rem cd ..
rem python preprocess.py training ..\data\d_dataset_p2 di --output_path=prep_data_13 --mfcc_coef=13
rem python preprocess.py training ..\data\d_dataset_p2 di --output_path=prep_data_20 --mfcc_coef=20

rem python training.py 73
rem python training.py 74
rem python training.py 75
rem python training.py 76
rem python training.py 1
rem python training.py 2
rem python training.py 3
rem python training.py 4
rem python training.py 5
rem python training.py 6
rem python training.py 7
rem python training.py 8
rem python training.py 9
rem python training.py 10
rem python training.py 11
rem python training.py 12
rem python training.py 13
rem python training.py 14
rem python training.py 15
rem python training.py 16
rem python training.py 17
rem python training.py 18
rem python training.py 19
rem python training.py 20
rem python training.py 21
rem python training.py 22
rem python training.py 23
rem python training.py 24
rem python training.py 25
rem python training.py 26
rem python training.py 27
rem python training.py 28
rem python training.py 29
rem python training.py 30
rem python training.py 31
rem python training.py 32
rem python training.py 33
rem python training.py 34
rem python training.py 35
rem python training.py 36
rem python training.py 37
rem python training.py 38
rem python training.py 39
rem python training.py 40
rem python training.py 41
rem python training.py 42
rem python training.py 43
rem python training.py 44
rem python training.py 45
rem python training.py 46
rem python training.py 47
rem python training.py 48
rem python training.py 49
rem python training.py 50
rem python training.py 51
rem python training.py 52
rem python training.py 53
rem python training.py 54
rem python training.py 55
rem python training.py 56
rem python training.py 57
rem python training.py 58
rem python training.py 59
rem python training.py 60
rem python training.py 61
rem python training.py 62
rem python training.py 63
rem python training.py 64

rem cd generator
rem python csv_to_wav.py assets\predefined_param.csv di 
rem python csv_to_wav.py assets\predefined_param.csv mono
rem cd ..
rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_20 --mfcc_coef=20
rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13 --mfcc_coef=13
rem python preprocess.py eval ..\data\m_eval mono --output_path=prep_data_20 --mfcc_coef=20
rem python preprocess.py eval ..\data\m_eval mono --output_path=prep_data_13 --mfcc_coef=13

rem cd generator
rem python generating.py
rem cd ..
rem python preprocess.py training ..\data\m_dataset_p2 mono --output_path=prep_data_13 --mfcc_coef=13
rem python preprocess.py training ..\data\m_dataset_p2 mono --output_path=prep_data_20 --mfcc_coef=20
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 22_model_baseline.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 13_model_with_embedded.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 14_model_with_pre_embedded.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 23_model_with_between_embedded.hdf5 di --label_normalize=3

rem python preprocess.py predict ..\data\d_records\d_record_set_1 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4
rem python preprocess.py predict ..\data\d_records\d_record_set_3 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4
rem python preprocess.py predict ..\data\d_records\d_record_set_4 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4
rem python preprocess.py predict ..\data\d_records\d_record_set_5 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4
rem python preprocess.py predict ..\data\d_records\d_record_set_6 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4
rem python preprocess.py predict ..\data\d_records\d_record_set_7 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4
rem python preprocess.py predict ..\data\d_records\d_record_set_8 di --output_path=prep_data_13 --resample_rate=16000 --feature_normalize=4

rem python predicting.py ..\data\d_records\d_record_set_1 prep_data_13 23_model_with_between_embedded.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_3 prep_data_13 23_model_with_between_embedded.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_4 prep_data_13 23_model_with_between_embedded.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_5 prep_data_13 23_model_with_between_embedded.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_6 prep_data_13 23_model_with_between_embedded.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_7 prep_data_13 23_model_with_between_embedded.hdf5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_8 prep_data_13 23_model_with_between_embedded.hdf5 di --label_normalize=3

rem python training.py 21
rem python training.py 13
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 21_senet_unem.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data 13_senet_unem.h5 di --label_normalize=3
rem python training.py 1
rem python training.py 2
rem python training.py 3
rem python training.py 4
rem python training.py 5
rem python training.py 6
rem python training.py 7
rem python training.py 8
rem python training.py 9
rem python training.py 10
rem python training.py 11
rem python training.py 12
rem python training.py 13
rem python training.py 14
rem python training.py 15
rem python training.py 16
rem python training.py 17
rem python training.py 18
rem python training.py 19
rem python training.py 20
rem python training.py 21
rem python training.py 22
rem python training.py 23
rem python training.py 24
rem python training.py 25
rem python training.py 26
rem python training.py 27
rem python training.py 28
rem python training.py 29
rem python training.py 30
rem python training.py 31
rem python training.py 32
rem python training.py 33
rem python training.py 34
rem python training.py 35
rem python training.py 36
rem python training.py 37
rem python training.py 38
rem python training.py 39
rem python training.py 40
rem python training.py 41
rem python training.py 42
rem python training.py 43
rem python training.py 44
rem python training.py 45
rem python training.py 46
rem python training.py 47
rem python training.py 48
rem python training.py 49
rem python training.py 50
rem python training.py 51
rem python training.py 52
rem python training.py 53
rem python training.py 54
rem python training.py 55
rem python training.py 56
rem python training.py 57
rem python training.py 58
rem python training.py 59
rem python training.py 60
rem python training.py 61
rem python training.py 62
rem python training.py 63
rem python training.py 64