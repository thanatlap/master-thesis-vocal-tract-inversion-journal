from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import os
from os.path import join
import tensorflow as tf
from tensorflow import keras


from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import regularizers
from tensorflow.keras import callbacks
from time import time
from datetime import datetime
import shutil 
import argparse

import lib.dev_utils as utils
import model as nn
import config as cf
import make_result as res

def prep_data():

	# load data from preprocess pipeline
	dataset = np.load(join(cf.DATASET_DIR,'training_subsets.npz'))
	X_train =dataset['X_train']
	y_train= dataset['y_train']
	X_val = dataset['X_val']
	y_val = dataset['y_val']
	X_test = dataset['X_test']
	y_test = dataset['y_test']

	print(X_train.shape)
	print(y_train.shape)

	if cf.CNN:
		X_train = utils.cnn_reshape(X_train)
		X_val = utils.cnn_reshape(X_val)
		X_test = utils.cnn_reshape(X_test)

	print('Train features and labels %s %s'%(str(X_train.shape),str(y_train.shape)))
	print('Validating features and labels %s %s'%(str(X_val.shape),str(y_val.shape)))
	print('Test features and labels %s %s'%(str(X_test.shape),str(y_test.shape)))

	return X_train, X_val, X_test, y_train, y_val, y_test

def get_model(model_fn, input_shape):
	'''
	This function return the model if the model is being load from save or checkpoint
	or initialize new model and weight
	'''
	# load from save if save is defined in config file
	if cf.LOAD_FROM_SAVE is not None:
		return tf.keras.models.load_model(cf.LOAD_FROM_SAVE, custom_objects={'rmse': nn.rmse})
	else:
		# initialize model
		model = model_fn(input_shape[0],input_shape[1])
		# load from weight if checkpoint is defined in config file
		if cf.LOAD_FROM_CHECKPOINT is not None:
			model.load_weights(cf.LOAD_FROM_CHECKPOINT)
		# inti optimizer and complied
		opt = optimizers.Adam(lr=cf.LEARNING_RATE, beta_1=cf.BETA1, beta_2=cf.BETA2, epsilon=cf.EPS, decay=cf.SDECAY, amsgrad=cf.AMSGRAD)
		model.compile(optimizer=opt,loss=cf.LOSS_FN,metrics=[nn.rmse])
	return model

def training(features, labels, val_features, val_labels, model, batch_size = cf.BATCH_SIZE, epochs = cf.EPOCHS, model_name=None, experiment_num=None):
	'''
	'This function perform model training'
	'''
	start = time()
	os.makedirs(join('model','checkpoint'), exist_ok=True)

	# Checkpoint
	checkpoint = callbacks.ModelCheckpoint(filepath=join('model','checkpoint','weights.{epoch:02d}-{val_rmse:.4f}.h5'), 
		monitor='val_rmse', verbose=1, mode='min',save_best_only=True, period = cf.CHECKPOINT_PEROID)
	
	if cf.EARLY_STOP_PATIENCE:
		# Early stop
		early = callbacks.EarlyStopping(monitor='val_loss', 
			min_delta=0, patience=cf.EARLY_STOP_PATIENCE, 
			verbose=1, mode='min', baseline=None, restore_best_weights=False)
		
		callback_list = [checkpoint, early]
	else:
		early = None
		callback_list = [checkpoint]

	history = model.fit(features,labels,
		batch_size=batch_size,
		epochs=epochs,
		validation_data=(val_features,val_labels),
		verbose=cf.MODEL_VERBOSE,
		callbacks=callback_list)

	total_time = time()-start
	print('[Total training time: %.3fs]'%total_time)
	if model_name:
		model_name = str(model_name)+'_'+str(datetime.now().strftime("%Y%m%d_%H%M"))+'.h5'
	else:
		model_name = 'model_'+str(datetime.now().strftime("%Y%m%d_%H%M"))+'.h5'

	if experiment_num:
		model_name = str(experiment_num)+'_'+model_name

	model.save(join('model',model_name))
	shutil.rmtree(join('model','checkpoint'))

	return history, total_time, early

def testing(features, labels, model):
	'''
	evaluate on test subset using rmse and R2
	'''
	y_pred, result = utils.evaluation_report(model,features,labels)
	r2 = utils.compute_AdjustR2(labels,y_pred,multioutput='uniform_average')
	return y_pred, result, r2

def training_fn(model_fn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=None, model_name=None):
	'''
	warpper function to training on each experiment
	This function is created for training multiple model in one run.
	'''
	# try:
	# load experiment number
	try:
		experiment_num = utils.get_experiment_number() if experiment_num is None else experiment_num
		print('Training experiment number: %s'%experiment_num)
		# initialize/load model
		model = get_model(model_fn = model_fn, input_shape = (X_train.shape[1],X_train.shape[2]))
		# training
		history, total_time, early = training(X_train, y_train, X_val, y_val, model, model_name=model_name, experiment_num=experiment_num)
		# evaluating
		y_pred, result, r2 = testing(X_test, y_test, model)
		# evaluate on training set
		training_y_pred, training_result, training_r2 = testing(X_train, y_train, model)
		# log experiment	
		res.log_result_train(experiment_num, X_train, X_val, X_test, y_train, y_val, y_test,y_pred, result, r2, history, model, 
			training_y_pred, training_result, training_r2, total_time, model_name, early)
		print('Result experiment number: #%s'%experiment_num)
		print('Result RMSE: %.4f'%(result[1]))
		print('Result R2: %.4f\n'%(r2))
	except Exception as e:
		print('experiment %f fail'%experiment_num)
		print(e)

def main(args):

	if args.exp in range(1,23):
		cf.DATASET_DIR = '../data/m_dataset_3_u_L/prep_data_exp7'
		cf.DI_SYLLABLE = False
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
	

	# -- MONOSYLLABLE MODEL

	

	# # -- DISYLLABLE MODEL
	if args.exp in range(23,100):
		cf.DATASET_DIR = '../data/d_dataset_3_u_L/prep_data_exp8'
		cf.DI_SYLLABLE = True
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()

	if args.exp == 57:
		cf.LOSS_FN = [nn.rmse]
		training_fn(nn.nn_cbf_5, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=188, model_name='nn_cbf_5')

	if args.exp == 58:
		cf.LOSS_FN = [nn.rmse]
		training_fn(nn.nn_cbf_4, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=189, model_name='nn_cbf_4')

	if args.exp == 59:
		cf.LOSS_FN = [nn.custom_loss5]
		training_fn(nn.nn_cbf_4, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=190, model_name='nn_cbf_4')

	if args.exp == 60:
		cf.LOSS_FN = [nn.custom_loss4]
		training_fn(nn.nn_cbf_4, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=191, model_name='nn_cbf_4')

	if args.exp == 61:
		cf.LOSS_FN = 'mse'
		cf.BATCH_SIZE = 128
		training_fn(nn.nn_cbf_4, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=192, model_name='nn_cbf_4')

	if args.exp == 62:
		cf.LOSS_FN = [nn.rmse]
		cf.BATCH_SIZE = 128
		training_fn(nn.nn_cbf_4, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=193, model_name='nn_cbf_4')

	if args.exp == 63:
		cf.LOSS_FN = 'mse'
		training_fn(nn.nn_cbf_4, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=206, model_name='nn_cbf_5')

	if args.exp == 64:
		cf.LOSS_FN = 'mse'
		training_fn(nn.nn_cbf_5, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=207, model_name='nn_cbf_5')

	if args.exp == 65:
		cf.LOSS_FN = 'mse'
		training_fn(nn.nn_bilstm_12, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=208, model_name='nn_bilstm_12')

	if args.exp == 66:
		cf.LOSS_FN = 'mse'
		training_fn(nn.resnet_1, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=209, model_name='resnet_1')

	if args.exp == 67:
		cf.LOSS_FN = 'mse'
		training_fn(nn.inception_1, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=210, model_name='inception_1')

	if args.exp == 68:
		cf.LOSS_FN = 'mse'
		training_fn(nn.inception_2, X_train, X_val, X_test, y_train, y_val, y_test, 
			experiment_num=211, model_name='inception_2')




if __name__ == '__main__':
	parser = argparse.ArgumentParser("Exp Control")
	parser.add_argument("exp", help="", type=int, default=0)
	args = parser.parse_args()
	main(args)