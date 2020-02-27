rem cd ../data_generator
rem python generating_data.py
rem cd ../experiment_0

rem python preprocess.py training ..\data\d_dataset_t3 di --output_path=aug_data --augment_samples=0.25 --resample_rate=16000 --label_normalize=5 --feature_normalize=1 --split_size=0.15
python training.py 6
rem python training.py 5

python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data --resample_rate=16000 --feature_normalize=1
