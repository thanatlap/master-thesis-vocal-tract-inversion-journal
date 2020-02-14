rem cd ..
rem cd data_generator
rem python generating_data.py
rem cd ../experiment_11

rem python preprocess.py training ..\data\d_dataset_t1 di --output_path=control_data --is_augment=False --resample_rate=16000 --label_normalize=5 --feature_normalize=3 --split_size=0.1
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
python training.py 11
python training.py 12
python training.py 13
python training.py 14
python training.py 15