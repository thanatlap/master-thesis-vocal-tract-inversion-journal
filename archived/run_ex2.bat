python preprocess.py training ..\data\d_dataset_p2 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_1 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_3 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_4 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_5 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_6 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_7 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_8 di --output_path=prep_data_13 --mfcc_coef=13

python preprocess.py training ..\data\m_dataset_p2 mono --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py eval ..\data\m_eval mono --output_path=prep_data_13 --mfcc_coef=13

python training.py 120
python training.py 121
python training.py 122
python training.py 123
python training.py 124
python training.py 125
python training.py 126
python training.py 127
python training.py 128
python training.py 129

python evaluating.py --model=120_baseline.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=121_FCNN.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=122_bilstm.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=123_LTRCNN.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=124_senet.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=125_baseline.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=126_FCNN.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=127_bilstm.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=128_LTRCNN.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=129_senet.h5 --prep=prep_data_13 --syllable=di

python predicting.py ..\data\d_records\d_record_set_1 prep_data_13 129_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 129_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_data_13 129_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_data_13 129_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_data_13 129_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_data_13 129_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_data_13 129_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_data_13 129_senet.h5 di --label_normalize=3

python predicting.py ..\data\d_records\d_record_set_1 prep_data_13 124_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 124_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_data_13 124_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_data_13 124_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_data_13 124_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_data_13 124_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_data_13 124_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_data_13 124_senet.h5 di --label_normalize=3

python training.py 130
python training.py 131

python evaluating.py --model=130_senet_em.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=131_senet_em.h5 --prep=prep_data_13 --syllable=mono
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 130_senet_em.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 131_senet_em.h5 di --label_normalize=3

cd generator
python generating.py
cd ..

python preprocess.py training ..\data\d_nospeaker_1 di --output_path=prep_data_13 --mfcc_coef=13
python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13_nss --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data_13_nss --mfcc_coef=13

python training.py 132
python evaluating.py --model=132_senet.h5 --prep=prep_data_13_nss --syllable=di
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13_nss 132_senet.h5 di --label_normalize=3


python preprocess.py training ..\data\d_nospeaker_1 di --output_path=prep_data_13_noaug --mfcc_coef=13 --is_augment=False
python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13_nss_na --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data_13_nss_na --mfcc_coef=13

python training.py 133
python evaluating.py --model=133_senet.h5 --prep=prep_data_13_nss_na --syllable=di
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13_nss_na 133_senet.h5 di --label_normalize=3

python training.py 127
python evaluating.py --model=127_bilstm.h5 --prep=prep_data_13 --syllable=di


python training.py 134
python evaluating.py --model=134_senet.h5 --prep=prep_data_13 --syllable=mono

python training.py 135
python evaluating.py --model=135_senet.h5 --prep=prep_data_13 --syllable=di
python predicting.py ..\data\d_records\d_record_set_1 prep_data_13 135_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 135_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_data_13 135_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_data_13 135_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_data_13 135_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_data_13 135_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_data_13 135_senet.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_data_13 135_senet.h5 di --label_normalize=3

python training.py 136
python evaluating.py --model=136_senet_em.h5 --prep=prep_data_13 --syllable=mono

python training.py 137
python evaluating.py --model=137_senet_em.h5 --prep=prep_data_13 --syllable=di
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 137_senet_em.h5 di --label_normalize=3

python training.py 138
python evaluating.py --model=138_senet.h5 --prep=prep_data_13_nss --syllable=di
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13_nss 138_senet.h5 di --label_normalize=3

python training.py 139
python evaluating.py --model=139_senet.h5 --prep=prep_data_13_nss_na --syllable=di
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13_nss_na 139_senet.h5 di --label_normalize=3

rem python preprocess.py training ..\data\d_nospeaker_1_1 di --output_path=prep_data_13 --mfcc_coef=13
rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13_nss --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data_13_nss --mfcc_coef=13

rem python preprocess.py training ..\data\d_nospeaker_1_1 di --output_path=prep_data_13_noaug --mfcc_coef=13 --is_augment=False
rem python preprocess.py eval ..\data\d_eval di --output_path=prep_data_13_nss_na --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data_13_nss_na --mfcc_coef=13


python evaluating.py --model=120_baseline.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=121_FCNN.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=122_bilstm.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=123_LTRCNN.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=124_senet.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=125_baseline.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=126_FCNN.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=127_bilstm.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=128_LTRCNN.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=129_senet.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=130_senet_em.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=131_senet_em.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=132_senet.h5 --prep=prep_data_13_nss --syllable=di
python evaluating.py --model=133_senet.h5 --prep=prep_data_13_nss_na --syllable=di
python evaluating.py --model=134_senet.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=135_senet.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=136_senet_em.h5 --prep=prep_data_13 --syllable=mono
python evaluating.py --model=137_senet_em.h5 --prep=prep_data_13 --syllable=di
python evaluating.py --model=138_senet.h5 --prep=prep_data_13_nss --syllable=di
python evaluating.py --model=139_senet.h5 --prep=prep_data_13_nss_na --syllable=di

rem --------------------------------------------------------------------------------------------------

rem python training.py 140
rem python evaluating.py --model=140_senet_v2.h5 --prep=prep_data_13 --syllable=di
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 140_senet_v2.h5 di --label_normalize=3

rem python training.py 141
rem python evaluating.py --model=141_senet_v2.h5 --prep=prep_data_13 --syllable=mono
rem python evaluating.py --model=125_baseline.h5 --prep=prep_data_13 --syllable=di

python predicting.py ..\data\d_records\d_record_set_1 prep_data_13 140_senet_v2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 140_senet_v2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_data_13 140_senet_v2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_data_13 140_senet_v2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_data_13 140_senet_v2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_data_13 140_senet_v2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_data_13 140_senet_v2.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_data_13 140_senet_v2.h5 di --label_normalize=3

python training.py 142
python evaluating.py --model=142_senet_em_v2.h5 --prep=prep_data_13 --syllable=mono

python training.py 143
python evaluating.py --model=143_senet_em_v2.h5 --prep=prep_data_13 --syllable=mono
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 143_senet_em_v2.h5 di --label_normalize=3

python training.py 144
python evaluating.py --model=144_senet_v2.h5 --prep=prep_data_13 --syllable=mono

python training.py 145
python evaluating.py --model=145_senet_v2.h5 --prep=prep_data_13 --syllable=mono
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 145_senet_v2.h5 di --label_normalize=3

rem python training.py 146
rem python evaluating.py --model=146_senet_v3.h5 --prep=prep_data_13 --syllable=di
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 146_senet_v3.h5 di --label_normalize=3

rem python training.py 147
rem python evaluating.py --model=147_senet_v3.h5 --prep=prep_data_13 --syllable=mono

rem python predicting.py ..\data\d_records\d_record_set_1 prep_data_13 146_senet_v3.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 146_senet_v3.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_3 prep_data_13 146_senet_v3.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_4 prep_data_13 146_senet_v3.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_5 prep_data_13 146_senet_v3.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_6 prep_data_13 146_senet_v3.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_7 prep_data_13 146_senet_v3.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_8 prep_data_13 146_senet_v3.h5 di --label_normalize=3

rem python training.py 148
rem python evaluating.py --model=148_senet_em_v3.h5 --prep=prep_data_13 --syllable=mono

rem python training.py 149
rem python evaluating.py --model=149_senet_em_v3.h5 --prep=prep_data_13 --syllable=mono
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 149_senet_em_v3.h5 di --label_normalize=3

rem python training.py 150
rem python evaluating.py --model=150_senet_v3.h5 --prep=prep_data_13 --syllable=di

rem python training.py 151
rem python evaluating.py --model=151_senet_v3.h5 --prep=prep_data_13 --syllable=di
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 151_senet_v3.h5 di --label_normalize=3

python predicting.py ..\data\d_records\d_record_set_2 prep_data_13 127_bilstm.h5 di --label_normalize=3

python predicting.py ..\data\d_records\d_record_set_1 prep_data_13_2 127_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13_2 127_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_data_13_2 127_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_data_13_2 127_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_data_13_2 127_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_data_13_2 127_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_data_13_2 127_bilstm.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_data_13_2 127_bilstm.h5 di --label_normalize=3

python predicting.py ..\data\d_records\d_record_set_1 prep_data_13_2 126_FCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13_2 126_FCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_data_13_2 126_FCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_data_13_2 126_FCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_data_13_2 126_FCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_data_13_2 126_FCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_data_13_2 126_FCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_data_13_2 126_FCNN.h5 di --label_normalize=3

python predicting.py ..\data\d_records\d_record_set_1 prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_2 prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_3 prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_4 prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_5 prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_6 prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_7 prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3
python predicting.py ..\data\d_records\d_record_set_8 prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3



rem python preprocess.py predict ..\data\d_records\d_record_set_1 di --output_path=prep_data_13_2 --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_2 di --output_path=prep_data_13_2 --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_3 di --output_path=prep_data_13_2 --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_4 di --output_path=prep_data_13_2 --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_5 di --output_path=prep_data_13_2 --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_6 di --output_path=prep_data_13_2 --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_7 di --output_path=prep_data_13_2 --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\d_record_set_8 di --output_path=prep_data_13_2 --mfcc_coef=13
rem python preprocess.py predict ..\data\d_records\selected di --output_path=prep_data_13_2 --mfcc_coef=13

rem python predicting.py ..\data\d_records\d_record_set_1 prep_data_13_2 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_2 prep_data_13_2 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_3 prep_data_13_2 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_4 prep_data_13_2 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_5 prep_data_13_2 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_6 prep_data_13_2 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_7 prep_data_13_2 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_set_8 prep_data_13_2 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\selected prep_data_13_2 140_senet_v2.h5 di --label_normalize=3

rem python preprocess.py predict ..\data\d_records\all di --output_path=prep_data_13_2 --mfcc_coef=13
rem python predicting.py ..\data\d_records\all prep_data_13_2 140_senet_v2.h5 di --label_normalize=3

rem python predicting.py ..\data\d_records\selected prep_data_13_2 144_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\selected prep_data_13_2 145_senet_v2.h5 di --label_normalize=3

rem python predicting.py ..\data\d_records\selected prep_data_13_2 127_bilstm.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\selected prep_data_13_2 128_LTRCNN.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\selected prep_data_13_2 126_FCNN.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\selected prep_data_13_2 125_baseline.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\selected prep_data_13_2 143_senet_em_v2.h5 di --label_normalize=3

rem python training.py 149
rem python predicting.py ..\data\d_records\selected prep_data_13_2 149_senet_v2.h5 di --label_normalize=3

rem python preprocess.py predict ..\data\d_records\all di --output_path=temp --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_aj di --output_path=prep_data_13 --mfcc_coef=13
python predicting.py ..\data\d_records\d_record_aj prep_data_13 140_senet_v2.h5 di --label_normalize=3
rem python predicting.py ..\data\d_records\d_record_aj prep_data_13 127_bilstm.h5 di --label_normalize=3

rem python preprocess.py training ..\data\d_dataset_p2 di --output_path=prep_data_13_v2 --mfcc_coef=13
rem python training.py 150
rem python preprocess.py predict ..\data\d_records\all di --output_path=temp --mfcc_coef=13
python preprocess.py predict ..\data\d_records\d_record_aj di --output_path=prep_data_13_v2 --mfcc_coef=13
python predicting.py ..\data\d_records\d_record_aj prep_data_13_v2 150_senet_v2.h5 di --label_normalize=3
rem python preprocess.py predict ..\data\d_records\selected di --output_path=prep_data_13_v2 --mfcc_coef=13
rem python predicting.py ..\data\d_records\selected prep_data_13_v2 150_senet_v2.h5 di --label_normalize=3

rem cd generator
rem python generating.py
rem cd ..


rem python preprocess.py training ..\data\d_dataset_3 di --output_path=prep_data_13_v2 --mfcc_coef=13
rem python training.py 152

rem python preprocess.py predict ..\data\d_records\d_record_aj di --output_path=prep_data_13_v7 --mfcc_coef=13
rem python predicting.py ..\data\d_records\d_record_aj prep_data_13_v7 151_senet_v2.h5 di --label_normalize=3

rem python preprocess.py predict ..\data\d_records\selected di --output_path=prep_data_13_v2 --mfcc_coef=13
rem python predicting.py ..\data\d_records\selected prep_data_13_v2 151_senet_v2.h5 di --label_normalize=3