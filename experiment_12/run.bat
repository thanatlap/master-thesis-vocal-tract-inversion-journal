rem cd ..
rem cd data_generator
rem python generating_data.py
rem cd ../experiment_11

rem python preprocess.py training ..\data\d_dataset_t1 di --output_path=control_data --is_augment=False --resample_rate=16000 --label_normalize=5 --feature_normalize=3 --split_size=0.1
python training.py 1
python training.py 2
python training.py 3
python training.py 4
