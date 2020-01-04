import numpy as np
import os
from os.path import join
import keras
from keras import models
import argparse
import model as nn
import lib.dev_utils as utils
import make_result as res
import lib.dev_gen as gen
import config as cf

def prep_data(data_dir, prep_folder, is_disyllable):
	'''
	import audio, represent in mfcc feature and normalize 
	'''
	return np.load(join(data_dir, prep_folder,'features.npy'))

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

	# model_file
	model_file = join('model',args.model_filename)
	if not os.path.exists(model_file):
		raise ValueError('Model %s does not exist!'%model_file)

	# prepare data
	features = prep_data(args.data_dir, args.prep_folder, args.syllable)
	# predict 
	y_pred = predict(features, model_file)
	# create output path
	if args.output_filename:
		# if output filename is specify
		output_path = join('result','predict_%s'%args.output_filename)
	else:
		# else, use default
		output_path = join('result','predict')
	os.makedirs(output_path, exist_ok=True)
	# read syllable name from text file
	with open(join(args.data_dir,'syllable_name.txt')) as f:
		syllable_name = np.array([word.strip() for line in f for word in line.split(',')])
		syllable_name = np.array([item for pair in syllable_name for item in pair]) if args.syllable else syllable_name
	# invert param back to predefined speaker scale
	params = utils.label_transform(y_pred, invert=True)
	# convert prediction result (monosyllabic) to disyllabic vowel
	params = gen.convert_to_disyllabic_parameter(params) if args.syllable else params
	
	# convert vocaltract parameter to audio
	gen.convert_param_to_wav(params, output_path, args.syllable)
	# load sound for comparison
	target_sound = np.array([join(args.data_dir, file) for file in np.load(join(args.data_dir, 'sound_set.npy'))])
	estimated_sound = np.array([join(output_path, 'sound', file) for file in np.load(join(output_path, 'npy', 'testset.npz'))['sound_sets']])
	# log result
	res.log_result_predict(y_pred, model_file, args.data_dir,output_path, target_sound, estimated_sound, syllable_name)
	# visualize spectrogram and wave plot
	utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(output_path,'spectrogram'), 'Greys')
	utils.generate_visualize_wav(target_sound, estimated_sound, join(output_path,'wave'))
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser("Predicting vocaltract from audio")
	parser.add_argument("data_dir", help="data directory (suppport npy)", type=str)
	parser.add_argument("prep_folder", help="folder contain preprocess data", type=str)
	parser.add_argument("model_filename", help="file of the model (hdf5)", type=str)
	parser.add_argument("syllable", help="is data disyllable or monosyllable ['mono','di']", type=str)
	parser.add_argument("--output_filename", help="a specific output filename for storing result", type=str, default=None)
	args = parser.parse_args()
	main(args)