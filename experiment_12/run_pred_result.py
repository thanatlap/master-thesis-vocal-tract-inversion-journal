import numpy as np
import os
from os.path import join
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
import argparse
import re
import model as nn
import lib.dev_utils as utils
import make_result as res
import lib.dev_gen as gen
import lib.dev_eval_result as evalresult

from functools import partial
np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

##### GLOBAL VARS #####

PARAM_1 = 'predict_1'
PARAM_2 = 'predict_2'
PARAM_3 = 'predict_3'
PARAM_4 = 'predict_4'
PARAM_5 = 'predict_5'

REC_DATA_1 = 'd_record_set_1'
REC_DATA_2 = 'd_record_set_2'
REC_DATA_3 = 'd_record_set_3'
REC_DATA_4 = 'd_record_set_4'
REC_DATA_5 = 'd_record_set_5'

DISYLLABLE = True

OUTPUT_FOLDER = 'predict_a1' 

# ======================= #

def main():

	params_1 = np.load(join('result',PARAM_1))
	params_2 = np.load(join('result',PARAM_2))
	params_3 = np.load(join('result',PARAM_3))
	params_4 = np.load(join('result',PARAM_4))
	params_5 = np.load(join('result',PARAM_5))

	# create output folder
	output_path = join('result', OUTPUT_FOLDER)
	os.makedirs(output_path)

	# Find mean of the param


	# The target sound must be average as well


	# convert vocaltract parameter to audio
	gen.convert_param_to_wav(params, output_path, is_disyllable, REC_DATA_1, mode='predict')
	# # load sound for comparison
	# with open(join(args.data_dir,'sound_set.txt'), 'r') as f:
	# 	files = np.array(f.read().split(','))

	# target_sound = gen.read_audio_path(args.data_dir)
	# estimated_sound = np.array([join(output_path, 'sound', file) for file in np.load(join(output_path, 'npy', 'testset.npz'))['sound_sets']])

	# # visualize spectrogram and wave plot
	# utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(output_path,'spectrogram'), 'Greys')
	# utils.generate_visualize_wav(target_sound, estimated_sound, join(output_path,'wave'))

	# # log result
	# res.log_result_predict(y_pred, model_file, args.data_dir,output_path, target_sound, estimated_sound, syllable_name)
	
	# evalresult.generate_eval_result(exp_num, is_disyllable, mode='predict', label_set=2)

if __name__ == '__main__':
	main()