import numpy as np
import pandas as pd
import os
from os.path import join
import keras
from keras import models

import model as nn
import config as cf
import lib.dev_utils as utils
import lib.dev_gen as gen
import make_result as res

def prep_data():
	'''
	import feature (represent in mfcc) and labels (scaled vocaltract parameter)
	'''
	# load data from preprocess pipeline
	dataset = np.load(join(cf.EVALSET_DIR,cf.PREP_EVAL_FOLDER,'eval_dataset.npz'))
	features =dataset['features']
	labels= dataset['labels']
	# delete WC from label because the model doesnt train to predict this
	labels = utils.delete_WC_param(labels)
	return features, labels

def evaluating(features, labels):
	'''
	Load model and evaluated the model using RMSE and R2
	'''
	#Load model for evaluated
	model_path = join('model', cf.MODEL_FILE)
	if not os.path.exists(model_path):
		raise ValueError('Model %s not found!'%model_path)

	model = models.load_model(model_path, custom_objects={'rmse': nn.rmse})
	model.summary()
	#Evaluate and predict
	eval_result = model.evaluate(features, labels)
	y_pred = model.predict(features)
	# compute r2
	r2 = utils.compute_R2(labels,y_pred,multioutput='uniform_average')
	return y_pred, eval_result, r2

def main():	

	experiment_num = utils.get_experiment_number()
	print('Experiment#%s on disyllable model: %s'%(experiment_num,str(cf.DI_SYLLABLE)))
	# import data
	features, labels = prep_data()
	# evaluated model using evalset
	y_pred, eval_result, r2 = evaluating(features, labels)
	print('Result Loss: %.4f\nRMSE: %.4f'%(eval_result[0],eval_result[1]))
	print('Result R2: %.4f'%(r2))
	# invert param back to predefined speaker scale
	params = utils.label_transform(y_pred, invert=True)
	# convert label into a sound if the model is D-AAI, the label is merge into disyllabic vowel
	# syllable name is not given because it already convert to disyllable since the prep_eval_set.py
	params = gen.convert_to_disyllabic_parameter(params) if cf.DI_SYLLABLE else params
	
	# get eval log path to store the result
	eval_dir = join('result', 'eval')
	# convert predicted vocaltract to audio
	gen.convert_param_to_wav(params, eval_dir, cf.DI_SYLLABLE)
	# load sound for comparison
	target_sound = np.array([join(cf.EVALSET_DIR, 'sound', file) for file in np.load(join(cf.EVALSET_DIR, 'npy', 'testset.npz'))['sound_sets']])
	estimated_sound = np.array([join(eval_dir, 'sound', file) for file in np.load(join(eval_dir, 'npy', 'testset.npz'))['sound_sets'] ])
	# log the result
	res.log_result_eval(labels, y_pred, eval_result, r2, target_sound, estimated_sound,experiment_num)
	# get visalization of spectrogram and wave plot
	utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(eval_dir,'spectrogram'), 'Greys')
	utils.generate_visualize_wav(target_sound, estimated_sound, join(eval_dir,'wave'))
	
if __name__ == '__main__':
	main()