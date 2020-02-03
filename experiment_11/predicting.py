import numpy as np
import os
from os.path import join
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks

from datetime import datetime
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

def prep_data(data_dir, prep_folder):
	'''
	import audio, represent in mfcc feature and normalize 
	'''
	features = np.load(join(data_dir, prep_folder,'features.npy'))
	print('Predict features %s'%str(features.shape))
	return features 

def predict(features, model_file):
	'''
	Load model and predict the vocaltract parameter from given audio
	'''
	#Load model for evaluated
	model = models.load_model(model_file, custom_objects={'rmse': nn.rmse})
	model.summary()
	y_pred = model.predict(features)
	return y_pred

def main(args):	

	if args.syllable.lower() not in ['mono', 'di']:
		raise ValueError('[ERROR] Syllable type %s is not define, support only ["mono", "di"]')

	is_disyllable = True if args.syllable.lower() == 'di' else False

	# model_file
	model_file = join('model',args.model_filename)
	if not os.path.exists(model_file):
		raise ValueError('[ERROR] Model %s does not exist!'%model_file)

	exp_num = int(re.search(r'\d+', args.model_filename).group())
	if exp_num is None:
		raise ValueError('[ERROR] Model filename %s does not contain exp_num!'%args.model_filename)

	# check label_normalize 
	if args.label_normalize not in [1,2, 3]:
		raise ValueError('[ERROR] Preprocess mode %s is not match [1: standardized, 2: min-max]'%args.label_normalize)

	# prepare data
	features = prep_data(args.data_dir, args.prep_folder)
	# predict 
	y_pred = predict(features, model_file)
	# create output path
	output_path = join('result','predict_%s'%exp_num)
	index = 1
	while os.path.exists(output_path):
		output_path = join('result','predict_%s_%s'%(exp_num,index))
		index += 1
	os.makedirs(output_path)
	# read syllable name from text file
	with open(join(args.data_dir,'syllable_name.txt')) as f:
		syllable_name = np.array([word.strip() for line in f for word in line.split(',')])
		syllable_name = np.array([ '%s;%s'%(item,str(idx+1)) for pair in syllable_name for idx, item in enumerate(pair)]) if is_disyllable else syllable_name

	if args.label_normalize == 1:
		params = utils.destandardized_label(y_pred, is_disyllable)
		params = utils.transform_VO(utils.add_params(params))
	elif args.label_normalize == 2:
		params = utils.descale_labels(y_pred)
		params = utils.transform_VO(utils.add_params(params))
	elif args.label_normalize == 3:
		params = utils.transform_VO(utils.add_params(y_pred))
		params = utils.min_max_descale_labels(params, is_disyllable)

	
	# convert prediction result (monosyllabic) to disyllabic vowel
	params = gen.convert_to_disyllabic_parameter(params) if is_disyllable else params
	# save param for averaging
	np.save(arr=params, file=join(output_path,'params.npy'))

	# convert vocaltract parameter to audio
	gen.convert_param_to_wav(params, output_path, is_disyllable, args.data_dir, mode='predict')
	# load sound for comparison
	with open(join(args.data_dir,'sound_set.txt'), 'r') as f:
		files = np.array(f.read().split(','))

	target_sound = gen.read_audio_path(args.data_dir)
	estimated_sound = np.array([join(output_path, 'sound', file) for file in np.load(join(output_path, 'npy', 'testset.npz'))['sound_sets']])

	# visualize spectrogram and wave plot
	utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(output_path,'spectrogram'), 'Greys')
	utils.generate_visualize_wav(target_sound, estimated_sound, join(output_path,'wave'))

	# log result
	res.log_result_predict(y_pred, model_file, args.data_dir,output_path, target_sound, estimated_sound, syllable_name)
	
	evalresult.generate_eval_result(exp_num, is_disyllable, mode='predict', label_set=2, output_path=join(output_path,'formant'))

	log = open(join(output_path,'description.txt'),"w")
	log.write('Date %s\n'%str(datetime.now().strftime("%Y-%B-%d %H:%M")))
	log.write('Data Dir: %s\n'%str(args.data_dir))
	log.write('Folder: %s\n'%str(args.prep_folder))
	log.write('Syllable: %s\n'%str(args.syllable))
	log.write('Model File: %s\n'%str(args.model_filename))
	log.write('Label normalize mode: %s\n'%str(args.label_normalize))
	log.close()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser("Predicting vocaltract from audio")
	parser.add_argument("data_dir", help="data directory (suppport npy)", type=str)
	parser.add_argument("prep_folder", help="folder contain preprocess data", type=str)
	parser.add_argument("model_filename", help="file of the model (hdf5)", type=str)
	parser.add_argument("syllable", help="is data disyllable or monosyllable ['mono','di']", type=str)
	parser.add_argument("--label_normalize", help="label normalize mode [1: standardized, 2: min-max, 3: min-max scaler]", type=int, default=1)
	args = parser.parse_args()
	main(args)