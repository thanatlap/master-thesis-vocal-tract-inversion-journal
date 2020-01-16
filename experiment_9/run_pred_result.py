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

def main(args):

	if args.syllable.lower() not in ['mono', 'di']:
		raise ValueError('[ERROR] Syllable type %s is not define, support only ["mono", "di"]')

	is_disyllable = True if args.syllable.lower() == 'di' else False

	params_1 = np.load(args.param_1)
	params_2 = np.load(args.param_2)
	params_3 = np.load(args.param_3)
	params_4 = np.load(args.param_4)
	params_5 = np.load(args.param_5)

	params = np.mean(params_1, params_2, params_3, params_4, params_5)

	# convert vocaltract parameter to audio
	gen.convert_param_to_wav(params, args.output_path, is_disyllable, args.data_dir, mode='predict')
	# load sound for comparison
	with open(join(args.data_dir,'sound_set.txt'), 'r') as f:
		files = np.array(f.read().split(','))

	target_sound = gen.read_audio_path(args.data_dir)
	estimated_sound = np.array([join(args.output_path, 'sound', file) for file in np.load(join(args.output_path, 'npy', 'testset.npz'))['sound_sets']])

	# visualize spectrogram and wave plot
	utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(args.output_path,'spectrogram'), 'Greys')
	utils.generate_visualize_wav(target_sound, estimated_sound, join(args.output_path,'wave'))

	# log result
	res.log_result_predict(y_pred, model_file, args.data_dir,args.output_path, target_sound, estimated_sound, syllable_name)
	
	evalresult.generate_eval_result(exp_num, is_disyllable, mode='predict', label_set=2)

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Make the final result")
	parser.add_argument("param_1", help="path to param", type=str)
	parser.add_argument("param_2", help="path to param", type=str)
	parser.add_argument("param_3", help="path to param", type=str)
	parser.add_argument("param_4", help="path to param", type=str)
	parser.add_argument("param_5", help="path to param", type=str)
	parser.add_argument("output_path", help="output folder of the final result", type=str)
	parser.add_argument("data_dir", help="data directory (choose one data)", type=str)
	parser.add_argument("syllable", help="is data disyllable or monosyllable ['mono','di']", type=str)
	args = parser.parse_args()
	main(args)