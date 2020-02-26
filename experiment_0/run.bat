cd ../data_generator
python generating_data.py
cd ../experiment_0

rem python preprocess.py training ..\data\d_dataset_t3 di --output_path=aug_data --augment_samples=0.3 --resample_rate=16000 --label_normalize=5 --feature_normalize=3 --split_size=0.1
rem python training.py 1