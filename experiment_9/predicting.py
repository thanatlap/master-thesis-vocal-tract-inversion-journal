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
	model = models.load_model(model_file, custom_objects={'rmse': nn.rmse, 
		'R2': nn.R2,
		'AdjustR2': nn.AdjustR2,
		'custom_loss':nn.custom_loss, 
		'custom_loss2':nn.custom_loss2,
		'custom_loss3':nn.custom_loss3,
		'custom_loss4':nn.custom_loss4,
		'custom_loss5':nn.custom_loss5,
		'custom_loss6':nn.custom_loss6,
		})
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
	if args.label_normalize not in [1,2]:
		raise ValueError('[ERROR] Preprocess mode %s is not match [1: standardized, 2: min-max]'%args.label_normalize)

	# prepare data
	features = prep_data(args.data_dir, args.prep_folder)
	# predict 
	y_pred = predict(features, model_file)
	# create output path
	if exp_num:
		# if output filename is specify
		output_path = join('result','predict_%s'%exp_num)
	else:
		# else, use default
		output_path = join('result','predict')
	os.makedirs(output_path, exist_ok=True)
	# read syllable name from text file
	with open(join(args.data_dir,'syllable_name.txt')) as f:
		syllable_name = np.array([word.strip() for line in f for word in line.split(',')])
		syllable_name = np.array([item for pair in syllable_name for item in pair]) if is_disyllable else syllable_name

	if args.label_normalize == 1:
		params = utils.destandardized_label(y_pred, is_disyllable)
	elif args.label_normalize == 2:
		params = utils.descale_labels(y_pred)

	# invert param back to predefined speaker scale
	params = utils.transform_VO(utils.add_params(params))
	# convert prediction result (monosyllabic) to disyllabic vowel
	params = gen.convert_to_disyllabic_parameter(params) if is_disyllable else params
	
	# convert vocaltract parameter to audio
	gen.convert_param_to_wav(params, output_path, is_disyllable)
	# load sound for comparison
	with open(join(args.data_dir,'sound_set.txt'), 'r') as f:
		files = np.array(f.read().split(','))
		
	target_sound = np.array([join(args.data_dir, file+'.wav') for file in files])
	estimated_sound = np.array([join(output_path, 'sound', file) for file in np.load(join(output_path, 'npy', 'testset.npz'))['sound_sets']])

	utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(output_path,'spectrogram'), 'Greys')
	utils.generate_visualize_wav(target_sound, estimated_sound, join(output_path,'wave'))

	# log result
	res.log_result_predict(y_pred, model_file, args.data_dir,output_path, target_sound, estimated_sound, syllable_name)
	# visualize spectrogram and wave plot

	
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser("Predicting vocaltract from audio")
	parser.add_argument("data_dir", help="data directory (suppport npy)", type=str)
	parser.add_argument("prep_folder", help="folder contain preprocess data", type=str)
	parser.add_argument("model_filename", help="file of the model (hdf5)", type=str)
	parser.add_argument("syllable", help="is data disyllable or monosyllable ['mono','di']", type=str)
	parser.add_argument("--label_normalize", help="label normalize mode [1: standardized, 2: min-max]", type=int, default=1)
	args = parser.parse_args()
	main(args)