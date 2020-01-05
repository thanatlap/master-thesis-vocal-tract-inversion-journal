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

import model as nn
import config as cf
import lib.dev_utils as utils
import lib.dev_gen as gen
import make_result as res
import lib.dev_eval_result as evalresult




def prep_data():
	'''
	import feature (represent in mfcc) and labels (scaled vocaltract parameter)
	'''
	# load data from preprocess pipeline
	dataset = np.load(join(cf.EVALSET_DIR,cf.PREP_EVAL_FOLDER,'eval_dataset.npz'))
	features =dataset['features']
	labels= dataset['labels']
	# delete WC from label because the model doesnt train to predict this
	labels = utils.delete_params(labels)

	if cf.CNN:
		features = utils.cnn_reshape(features)

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

	model = models.load_model(model_path, custom_objects={'rmse': nn.rmse, 
		'R2': nn.R2,
		'AdjustR2': nn.AdjustR2,
		'custom_loss':nn.custom_loss, 
		'custom_loss2':nn.custom_loss2,
		'custom_loss3':nn.custom_loss3,
		'custom_loss4':nn.custom_loss4,
		'custom_loss5':nn.custom_loss5,
		'custom_loss6':nn.custom_loss6,
		'custom_loss7':nn.custom_loss7,
		})
	model.summary()
	#Evaluate and predict
	eval_result = model.evaluate(features, labels)
	y_pred = model.predict(features)
	# compute r2
	r2 = utils.compute_R2(labels,y_pred,multioutput='uniform_average')
	return y_pred, eval_result, r2

def main():	

	print('Experiment#%s on disyllable model: %s'%(cf.EVAL_EXP_NUM,str(cf.DI_SYLLABLE)))
	# import data
	features, labels = prep_data()
	# evaluated model using evalset
	y_pred, eval_result, r2 = evaluating(features, labels)
	print('Result Loss: %.4f\nRMSE: %.4f'%(eval_result[0],eval_result[1]))
	print('Result R2: %.4f'%(r2))
	# invert param back to predefined speaker scale
	print('[INFO] transform label')
	params = utils.label_transform(y_pred, invert=True)
	# convert label into a sound if the model is D-AAI, the label is merge into disyllabic vowel
	# syllable name is not given because it already convert to disyllable since the prep_eval_set.py
	params = gen.convert_to_disyllabic_parameter(params) if cf.DI_SYLLABLE else params
	
	# get eval log path to store the result
	eval_dir = join('result', 'eval_'+str(cf.EVAL_EXP_NUM) )
	# convert predicted vocaltract to audio
	print('[INFO] generated wav')
	gen.convert_param_to_wav(params, eval_dir, cf.DI_SYLLABLE)
	# load sound for comparison
	target_sound = np.array([join(cf.EVALSET_DIR, 'sound', file) for file in np.load(join(cf.EVALSET_DIR, 'npy', 'testset.npz'))['sound_sets']])
	estimated_sound = np.array([join(eval_dir, 'sound', file) for file in np.load(join(eval_dir, 'npy', 'testset.npz'))['sound_sets'] ])
	# log the result
	print('[INFO] logging result')
	res.log_result_eval(labels, y_pred, eval_result, r2, target_sound, estimated_sound,cf.EVAL_EXP_NUM)
	# get visalization of spectrogram and wave plot
	print('[INFO] generate spectrogram')
	utils.generate_visualize_spectrogram(target_sound, estimated_sound, join(eval_dir,'spectrogram'), 'Greys')

	print('[INFO] generate wav')
	utils.generate_visualize_wav(target_sound, estimated_sound, join(eval_dir,'wave'))

	print('[INFO] generate evaluation result')
	evalresult.generate_eval_result(cf.EVAL_EXP_NUM, cf.DI_SYLLABLE)


if __name__ == '__main__':
	main()