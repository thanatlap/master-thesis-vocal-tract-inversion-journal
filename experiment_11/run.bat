rem cd ..
rem cd data_generator
rem python generating_data.py
rem cd ../experiment_11

rem python preprocess.py training ..\data\d_dataset_t1 di --output_path=control_data --is_augment=False --resample_rate=16000 --label_normalize=5 --feature_normalize=3 --split_size=0.1
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
python training.py 22
python training.py 23
python training.py 24
python training.py 25
python training.py 26
python training.py 27
python training.py 28
python training.py 29
python training.py 30
python training.py 31
python training.py 32
python training.py 33
python training.py 34
python training.py 35
python training.py 36
python training.py 37
python training.py 38
python training.py 39
python training.py 40
python training.py 41
python training.py 42
python training.py 43
python training.py 44
python training.py 45
python training.py 46
python training.py 47
python training.py 48
python training.py 49
python training.py 50
python training.py 51
python training.py 52
python training.py 53
python training.py 54
python training.py 55
python training.py 56
python training.py 57
python training.py 58
python training.py 59
python training.py 60
python training.py 61
python training.py 62
python training.py 63
python training.py 64
python training.py 65
python training.py 66
python training.py 67
python training.py 68
python training.py 69
python training.py 70
python training.py 71
python training.py 72
python training.py 73
python training.py 74
python training.py 75
python training.py 76
python training.py 77
python training.py 78
rem python training.py 79
rem python training.py 80
rem python training.py 81
rem python training.py 82
rem python training.py 83
rem python training.py 84
rem python training.py 85
rem python training.py 86
rem python training.py 87
rem python training.py 88
rem python training.py 89
rem python training.py 90
rem python training.py 91
rem python training.py 92
rem python training.py 93
rem python training.py 94
rem python training.py 95
rem python training.py 96
rem python training.py 97
rem python training.py 98
rem python training.py 99
rem python training.py 100
rem python preprocess.py training ..\data\d_dataset_t1 di --output_path=aug_data --augment_samples=0.3 --resample_rate=16000 --label_normalize=5 --feature_normalize=3 --split_size=0.1
rem python training.py 101
rem python training.py 102
rem python training.py 103
rem python training.py 104
rem python training.py 105
rem python training.py 106
rem python training.py 107
rem python training.py 108

rem cd ..
rem cd data_generator
rem python generating_data.py
rem cd ../experiment_11

rem python preprocess.py training ..\data\m_dataset_t1 mono --output_path=aug_data --augment_samples=0.3 --resample_rate=16000 --label_normalize=5 --feature_normalize=3 --split_size=0.1
rem python preprocess.py training ..\data\m_dataset_t1 mono --output_path=control_data --is_augment=False --resample_rate=16000 --label_normalize=5 --feature_normalize=3 --split_size=0.1

