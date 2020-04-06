import numpy as np
import pandas as pd
import os
from os.path import join
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
import re
import argparse
from functools import partial

import model as nn
import config as cf
import lib.dev_utils as utils
import lib.dev_gen as gen
import make_result as res
import lib.dev_eval_result as evalresult

np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


def prep_data():
	'''
	import feature (represent in mfcc) and labels (scaled vocaltract parameter)
	'''
	# load data from preprocess pipeline
	dataset = np.load(join(cf.EVALSET_DIR,cf.PREP_EVAL_FOLDER,'eval_dataset.npz'))
	features =dataset['features']
	labels= dataset['labels']

	labels = utils.delete_params(labels)

	print('Eval features and labels %s %s'%(str(features.shape),str(labels.shape)))

	return features, labels

def evaluating(features, labels):
	'''
	Load model and evaluated the model using RMSE and R2
	'''
	#Load model for evaluated
	model_path = join('model', cf.MODEL_FILE)
	if not os.path.exists(model_path):
		raise ValueError('Model %s not found!'%model_path)

	model = models.load_model(model_path, custom_objects={'rmse': nn.rmse, 'R2':nn.R2})
	model.summary()
	#Evaluate and predict
	eval_result = model.evaluate(features, labels)
	y_pred = model.predict(features)
	# compute r2
	r2 = utils.compute_R2(labels,y_pred,multioutput='uniform_average')
	return y_pred, eval_result, r2

def main(args):	

	if args.model:
		cf.MODEL_FILE = args.model

	if args.prep:
		cf.PREP_EVAL_FOLDER = args.prep

	if args.syllable:
		if args.syllable == 'mono':
			cf.DI_SYLLABLE = False
			cf.EVALSET_DIR = '../data/m_eval'
		else:
			cf.DI_SYLLABLE = True
			cf.EVALSET_DIR = '../data/d_eval'

	print(cf.PREP_EVAL_FOLDER)
	print(cf.DI_SYLLABLE)
	print(cf.EVALSET_DIR)

	print('[DEBUG] Model in used: {}'.format(cf.MODEL_FILE))

	exp_num = int(re.search(r'\d+', cf.MODEL_FILE).group())
	if exp_num is None:
		raise ValueError('[ERROR] Model filename %s does not contain exp_num!'%args.model_filename)

	print('Experiment#%s on disyllable model: %s'%(exp_num,str(cf.DI_SYLLABLE)))
	# import data
	features, labels = prep_data()
	# evaluated model using evalset
	y_pred, eval_result, r2 = evaluating(features, labels)
	print('Result LOSS: %.3f RMSE: %.3f R2:%.3f'%(eval_result[0],eval_result[1],eval_result[2]))
	# invert param back to predefined speaker scale
	print('[INFO] transform label')

	params = utils.detransform_label(cf.LABEL_MODE, y_pred, cf.DI_SYLLABLE)

	# convert label into a sound if the model is D-AAI, the label is merge into disyllabic vowel
	# syllable name is not given because it already convert to disyllable since the prep_eval_set.py
	params = gen.convert_to_disyllabic_parameter(params) if cf.DI_SYLLABLE else params
	
	# get eval log path to store the result
	eval_dir = join('result', 'eval_'+str(exp_num) )
	# convert predicted vocaltract to audio
	print('[INFO] generated wav')
	gen.convert_param_to_wav(params, eval_dir, cf.DI_SYLLABLE)
	# load sound for comparison
	target_sound = np.array([join(cf.EVALSET_DIR, 'sound',file) for file in np.load(join(cf.EVALSET_DIR, 'csv_dataset.npz'))['sound_sets'][0]])
	estimated_sound = np.array([join(eval_dir, 'sound', file) for file in np.load(join(eval_dir, 'testset.npz'))['sound_sets'] ])

	# log the result
	print('[INFO] logging result')
	res.log_result_eval(labels, y_pred, eval_result, r2, target_sound, estimated_sound,exp_num)
	# get visalization of spectrogram and wave plot
	print('[INFO] generate spectrogram')
	utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(eval_dir,'spectrogram'), 'Greys')

	print('[INFO] generate wav')
	utils.generate_visualize_wav(target_sound, estimated_sound, join(eval_dir,'wave'))

	print('[INFO] generate evaluation result')
	evalresult.generate_eval_result(exp_num, cf.DI_SYLLABLE)


if __name__ == '__main__':
	parser = argparse.ArgumentParser("Model to eval")
	parser.add_argument("--model", help="file of the model (hdf5)", type=str, default=None)
	parser.add_argument("--prep", help="file of the model (hdf5)", type=str, default=None)
	parser.add_argument("--syllable", help="[mono, di]", type=str, default=None)
	args = parser.parse_args()
	main(args)