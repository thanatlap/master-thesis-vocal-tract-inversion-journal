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

	# preprocess subset by deleted WC param (because it alway 0)
	y_train = utils.delete_params(y_train)
	y_val = utils.delete_params(y_val)
	y_test = utils.delete_params(y_test)

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

def training(features, labels, val_features, val_labels, model, batch_size = cf.BATCH_SIZE, epochs = cf.EPOCHS, model_name=None):
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
			verbose=1, mode='min', baseline=0.050, restore_best_weights=False)
		
		callback_list = [checkpoint, early]
	else:
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
	model.save(join('model',model_name))
	shutil.rmtree(join('model','checkpoint'))

	return history, total_time

def testing(features, labels, model):
	'''
	evaluate on test subset using rmse and R2
	'''
	y_pred, result = utils.evaluation_report(model,features,labels)
	r2 = utils.compute_R2(labels,y_pred,multioutput='uniform_average')
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
		history, total_time = training(X_train, y_train, X_val, y_val, model, model_name=model_name)
		# evaluating
		y_pred, result, r2 = testing(X_test, y_test, model)
		# evaluate on training set
		training_y_pred, training_result, training_r2 = testing(X_train, y_train, model)
		# log experiment	
		res.log_result_train(experiment_num, X_train, X_val, X_test, y_train, y_val, y_test,y_pred, result, r2, history, model, 
			training_y_pred, training_result, training_r2, total_time, model_name)
		print('Result experiment number: #%s'%experiment_num)
		print('Result RMSE: %.4f'%(result[1]))
		print('Result R2: %.4f\n'%(r2))
	except Exception as e:
		print('experiment %f fail'%experiment_num)
		print(e)

def main():

	X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
	# -- DISYLLABLE MODEL

	# -- MONOSYLLABLE MODEL

	# train with AMSGrad
	# training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=37, model_name='bilstm')
	# training_fn(nn.nn_bilstm_12, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=38, model_name='bilstm_12')
	# training_fn(nn.nn_bilstm_13, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=39, model_name='bilstm_13')

	# training_fn(nn.nn_bilstm_14, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=40, model_name='bilstm_14')
	# training_fn(nn.nn_bilstm_15, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=41, model_name='bilstm_15')
	# training_fn(nn.nn_bilstm_16, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=42, model_name='bilstm_16')
	# training_fn(nn.nn_bilstm_17, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=43, model_name='bilstm_17')
	# training_fn(nn.nn_bilstm_18, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=44, model_name='bilstm_18')
	# training_fn(nn.nn_bilstm_19, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=45, model_name='bilstm_19')
	# training_fn(nn.nn_bilstm_20, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=46, model_name='bilstm_20')
	# training_fn(nn.nn_bilstm_21, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=47, model_name='bilstm_21')
	# training_fn(nn.nn_fc_bilstm_fc, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=48, model_name='fc_bilstm_fc')
	# training_fn(nn.nn_fc_bilstm_fc_drop, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=49, model_name='fc_bilstm_fc_drop')
	# training_fn(nn.nn_fc_bilstm_fc_2, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=50, model_name='fc_bilstm_fc_2')
	# training_fn(nn.nn_fc_bilstm_fc_2_drop, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=51, model_name='fc_bilstm_fc_2_drop')
	# training_fn(nn.nn_fc_bilstm_cn_fc_drop, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=52, model_name='fc_bilstm_cnn_fc_drop')
	training_fn(nn.nn_fc_bilstm_fc_drop, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=53, model_name='fc_bilstm_fc_drop_m3')
	training_fn(nn.nn_fc_bilstm_fc_2_drop, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=54, model_name='fc_bilstm_fc_2_drop_m3')
	training_fn(nn.nn_fc_bilstm_cn_fc_drop, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=55, model_name='fc_bilstm_cnn_fc_drop_m3')
	

if __name__ == '__main__':
	main()