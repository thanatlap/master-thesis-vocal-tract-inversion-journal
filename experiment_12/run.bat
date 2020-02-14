cd ..
cd data_generator
python generating_data.py
cd ../experiment_12

python preprocess.py training ..\data\d_dataset_t2 di --output_path=aug_data --augment_samples=0.3 --resample_rate=16000 --label_normalize=5 --feature_normalize=3 --split_size=0.1

python training.py 1
python training.py 2
python training.py 3
python training.py 4
python training.py 5
python training.py 6
python training.py 7
python training.py 8
python training.py 9
python training.py 10
python training.py 11
python training.py 12
python training.py 13
python training.py 14
python training.py 15
python training.py 16
python training.py 17
python training.py 18
python training.py 19
python training.py 20
python training.py 21