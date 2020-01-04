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
		# opt = optimizers.Adam(lr=cf.LEARNING_RATE, beta_1=cf.BETA1, beta_2=cf.BETA2, epsilon=cf.EPS, decay=cf.SDECAY, amsgrad=cf.AMSGRAD)
		opt = optimizers.RMSprop()
		model.compile(optimizer=opt,loss=cf.LOSS_FN,metrics=[nn.rmse])
	return model

def training(features, labels, val_features, val_labels, model, batch_size = cf.BATCH_SIZE, epochs = cf.EPOCHS):
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
	model.save(join('model','model_'+str(datetime.now().strftime("%Y%m%d_%H%M"))+'.h5'))
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
		history, total_time = training(X_train, y_train, X_val, y_val, model)
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
	except:
		print('experiment %f fail'%experiment_num)

def main():

	X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
	# -- DISYLLABLE MODEL
	# training_fn(nn.nn_fc, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=1)
	# training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=2)
	# training_fn(nn.nn_lstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=3)
	#training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=9)
	
	# training_fn(nn.nn_bilstm_3, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=11)
	# training_fn(nn.nn_bilstm_4, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=12)
	# training_fn(nn.nn_bilstm_5, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=13)
	# training_fn(nn.nn_bilstm_6, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=14)
	# training_fn(nn.nn_bilstm_7, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=15)
	# training_fn(nn.nn_bilstm_8, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=16)
	# training_fn(nn.nn_bilstm_9, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=17)
	# training_fn(nn.nn_bilstm_10, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=18)
	# training_fn(nn.nn_fc_bn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=29)


	# # -- DISYLLABLE MODEL CNN
	# cf.CNN = True
	# X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
	# training_fn(nn.nn_cnn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=4)

	# cf.EARLY_STOP_PATIENCE = 10
	# training_fn(nn.nn_bilstm_11, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=31)
	# training_fn(nn.nn_bilstm_12, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=35)

	# training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=37) # with RMSProp
	# training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=39) # with RMSProp

	# -- MONOSYLLABLE MODEL

	cf.DI_SYLLABLE = False
	cf.DATASET_DIR = '../data/m_dataset_1/prep_data'
	X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
	# #training_fn(nn.nn_fc, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=5)
	# # training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=6)
	# # training_fn(nn.nn_lstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=7)
	# training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=10)
	# training_fn(nn.nn_fc_bn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=30)
	# training_fn(nn.nn_fc_bn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=38) # with RMSProp
	# training_fn(nn.nn_fc_bn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=40) # with RMSProp
	training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=41, model_name='nn_bilstm') # with RMSProp
	
	# cf.EARLY_STOP_PATIENCE = 10
	# training_fn(nn.nn_fc_bn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=28)
	# training_fn(nn.nn_bilstm_11, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=32)
	# training_fn(nn.nn_bilstm_12, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=36)
	# -- MONOSYLLABLE MODEL CNN
	# cf.CNN = True
	# X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
	# training_fn(nn.nn_cnn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=8)


	# cf.DI_SYLLABLE = True
	# cf.DATASET_DIR = '../data/d_dataset_2/prep_data'
	# X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
	# cf.EARLY_STOP_PATIENCE = 10
	# training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=19)
	# training_fn(nn.nn_bilstm_3, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=20)
	# training_fn(nn.nn_bilstm_4, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=21)
	# training_fn(nn.nn_bilstm_5, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=22)
	# training_fn(nn.nn_bilstm_6, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=23)
	# training_fn(nn.nn_bilstm_7, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=24)
	# training_fn(nn.nn_bilstm_8, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=25)
	# training_fn(nn.nn_bilstm_9, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=26)
	# training_fn(nn.nn_bilstm_10, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=27)

	# ===== USE RMSE as a LOSS FUNCTION ======
	# not different from using MSE
	# cf.EARLY_STOP_PATIENCE = 10
	# cf.LOSS_FN = [nn.rmse]
	# training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=33)

	# cf.DI_SYLLABLE = False
	# cf.DATASET_DIR = '../data/m_dataset_1/prep_data'
	# X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
	# training_fn(nn.nn_bilstm, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=34)




if __name__ == '__main__':
	main()