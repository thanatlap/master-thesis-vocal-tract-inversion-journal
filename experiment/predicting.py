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
# import generator.gen_tools as gen
import lib.dev_gen as gen
import lib.dev_eval_result as evalresult

from functools import partial

np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def import_data(data_dir, prep_folder, is_disyllable):

	features = np.load(join(data_dir, prep_folder,'features.npy')) 
	syllable_name = utils.read_syllable_from_txt(args.data_dir, is_disyllable)

	return features, syllable_name

def predict(features, model_file):

	#Load model for evaluated
	model = models.load_model(model_file, custom_objects={'rmse': nn.rmse, 'R2': nn.R2})
	model.summary()
	return model.predict(features)

def main(args):	

	if args.syllable.lower() not in ['mono', 'di']:
		raise ValueError('[ERROR] Syllable {} is not supported, accepted only ["mono", "di"]'.format(args.syllable))
	else:
		is_disyllable = True if args.syllable.lower() == 'di' else False

	# model_file
	model_file = join('model', args.model_filename)
	if not os.path.exists(model_file):
		raise ValueError('[ERROR] Model {} does not exist!'.format(model_file))

	# extract experiment num from model file
	exp_num = int(re.search(r'\d+', args.model_filename).group())
	if exp_num is None:
		raise ValueError('[ERROR] Model filename {} does not contain exp_num!'.format(args.model_filename))

	# prepare data
	features, syllable_name = import_data(args.data_dir, args.prep_folder, is_disyllable)
	# predict 
	y_pred = predict(features, model_file)

	# create output path
	output_path = utils.created_file_path_predict(exp_num)

	params = utils.detransform_label(y_pred, is_disyllable)
	
	# convert prediction result (monosyllabic) to disyllabic vowel
	params = gen.convert_to_disyllabic_parameter(params) if is_disyllable else params
	
	# convert vocaltract parameter to audio
	gen.convert_param_to_wav(params, output_path, is_disyllable, args.data_dir, mode='predict')

	target_sound = gen.read_audio_path(args.data_dir)
	estimated_sound = np.array([join(output_path, 'sound', file) for file in np.load(join(output_path,'testset.npz'))['sound_sets']])

	# log result
	if is_disyllable: gen.average_parameter_vocaltract(params, syllable_name, output_path)
	res.log_result_predict(y_pred, model_file, args.data_dir,output_path, target_sound, estimated_sound, syllable_name)	
	evalresult.generate_eval_result(exp_num, is_disyllable, mode='predict', label_set=2, output_path=join(output_path,'formant'))
	utils.pca_articulation_plot(y_pred.reshape(y_pred.shape[0]//2,2,y_pred.shape[1]), args.data_dir, output_path)
	# utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(output_path,'spectrogram'), 'Greys')
	# utils.generate_visualize_wav(target_sound, estimated_sound, join(output_path,'wave'))

	# export 
	np.save(arr=params, file=join(output_path,'predict_params.npy'))

	log = open(join(output_path,'description.txt'),"w")
	log.write('Date %s\n'%str(datetime.now().strftime("%Y-%B-%d %H:%M")))
	log.write('Data Dir: %s\n'%str(args.data_dir))
	log.write('Folder: %s\n'%str(args.prep_folder))
	log.write('Syllable: %s\n'%str(args.syllable))
	log.write('Model File: %s\n'%str(args.model_filename))
	log.close()
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser("Predicting vocaltract from audio")
	parser.add_argument("data_dir", help="data directory (suppport npy)", type=str)
	parser.add_argument("prep_folder", help="folder contain preprocess data", type=str)
	parser.add_argument("model_filename", help="file of the model (hdf5)", type=str)
	parser.add_argument("syllable", help="is data disyllable or monosyllable ['mono','di']", type=str)
	args = parser.parse_args()
	main(args)