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
from functools import partial
import pickle

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

	y_train = utils.delete_params(y_train)
	y_val = utils.delete_params(y_val)
	y_test = utils.delete_params(y_test)

	# if cf.CNN:
	# 	X_train = utils.cnn_reshape(X_train)
	# 	X_val = utils.cnn_reshape(X_val)
	# 	X_test = utils.cnn_reshape(X_test)

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
		return tf.keras.models.load_model(join('model',cf.LOAD_FROM_SAVE), custom_objects={'rmse': nn.rmse})
	else:
		# initialize model
		model = model_fn(input_shape[0],input_shape[1])
		# load from weight if checkpoint is defined in config file
		if cf.LOAD_FROM_CHECKPOINT is not None:
			print('[INFO] Continue from checkpoint')
			model.load_weights(join('model','checkpoint',cf.LOAD_FROM_CHECKPOINT))
		# inti optimizer and complied
		if cf.OPT_NUM == 1:
			opt = optimizers.Adam(lr=cf.LEARNING_RATE, beta_1=cf.BETA1, beta_2=cf.BETA2, epsilon=cf.EPS, decay=cf.SDECAY, amsgrad=cf.AMSGRAD)
		elif cf.OPT_NUM == 2:
			opt = optimizers.Nadam()
		elif cf.OPT_NUM == 3:
			opt = optimizers.RMSprop(learning_rate = cf.LEARNING_RATE)
		
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

	if cf.TENSORBOARD:
		log_dir = join('tf_log','log_'+str(experiment_num)+str(datetime.now().strftime("%Y%m%d-%H%M%S")))
		os.makedirs(log_dir, exist_ok=True)
		tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
		callback_list.append(tensorboard_callback)

	history = model.fit(features,labels,
		batch_size=batch_size,
		epochs=epochs,
		validation_data=(val_features,val_labels),
		verbose=cf.MODEL_VERBOSE,
		callbacks=callback_list)

	total_time = time()-start
	print('[Total training time: %.3fs]'%total_time)
	if model_name:
		# model_name = str(model_name)+'_'+str(datetime.now().strftime("%Y%m%d_%H%M"))+'.h5'
		model_name = str(model_name)+'.h5'
	else:
		model_name = 'model_'+str(datetime.now().strftime("%Y%m%d_%H%M"))+'.h5'

	if experiment_num:
		model_name = str(experiment_num)+'_'+model_name

	model.save(join('model',model_name))
	with open(join('model','hist_'+model_name[:-3]), 'wb') as file:
		pickle.dump(history.history, file)
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
	# load experiment number
	try:
		experiment_num = utils.get_experiment_number() if experiment_num is None else experiment_num
		print('### --- Training experiment number: %s'%experiment_num)
		# initialize/load model
		model = get_model(model_fn = model_fn, input_shape = (X_train.shape[1],X_train.shape[2]))
		# training
		history, total_time, early = training(X_train, y_train, X_val, y_val, model, model_name=model_name, experiment_num=experiment_num)
		# evaluating
		y_pred, result, r2 = testing(X_test, y_test, model)
		# evaluate on training set
		training_y_pred, training_result, training_r2 = testing(X_train, y_train, model)
		val_y_pred, val_result, val_r2 = testing(X_val, y_val, model)
		print('Result experiment number: #%s'%experiment_num)
		print('Result RMSE: %.4f'%(result[1]))
		print('Result R2: %.4f\n'%(r2))
		print('[INFO] Log Results')
		# log experiment	
		res.log_result_train(experiment_num, X_train, X_val, X_test, y_train, y_val, y_test,y_pred, result, r2, history, model, 
			training_y_pred, training_result, training_r2, total_time, model_name, early, val_y_pred, val_result, val_r2)
		
	except Exception as e:
		print('experiment %f fail'%experiment_num)
		print(e)

def main(args):

	X_train, X_val, X_test, y_train, y_val, y_test = prep_data()

	ptraining_fn = partial(training_fn, 
		X_train=X_train, 
		X_val=X_val, 
		X_test=X_test, 
		y_train=y_train, 
		y_val=y_val, 
		y_test=y_test,
		experiment_num=0, 
		model_name='undefined')


	# dropout = None
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.001 
	if args.exp == 1: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 2: ptraining_fn(nn.init_FCNN(), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 3: ptraining_fn(nn.inti_lstm(), experiment_num=args.exp, model_name='lstm')
	if args.exp == 4: ptraining_fn(nn.inti_gru(), experiment_num=args.exp, model_name='gru')
	if args.exp == 5: ptraining_fn(nn.inti_bilstm(), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 6: ptraining_fn(nn.inti_bigru(), experiment_num=args.exp, model_name='bigru')
	if args.exp == 7: ptraining_fn(nn.inti_cnn_bilstm(), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 8: ptraining_fn(nn.inti_cnn_bigru(), experiment_num=args.exp, model_name='cnn_bigru')


	# dropout = 0.3
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 9: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 10: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 11: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 12: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 13: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 14: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 15: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')


	# dropout = 0.5
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.5
	if args.exp == 16: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 17: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 18: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 19: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 20: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 21: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 22: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')


	# batch size = 16
	cf.BATCH_SIZE = 16
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 23: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 24: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 25: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 26: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 27: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 28: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 29: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 30: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')

	# batch size = 32
	cf.BATCH_SIZE = 32
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 31: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 32: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 33: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 34: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 35: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 36: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 37: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 38: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')


	# batch size = 128
	cf.BATCH_SIZE = 128
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 39: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 40: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 41: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 42: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 43: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 44: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 45: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 46: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')


	# learning rate = 0.01
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.01 
	DROP_RATE = 0.3
	if args.exp == 47: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 48: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 49: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 50: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 51: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 52: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 53: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 54: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')


	# learning rate = 0.0001
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.0001 
	DROP_RATE = 0.3
	if args.exp == 55: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 56: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 57: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 58: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 59: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 60: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 61: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 62: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')

	# patience = 5
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 1000
	cf.EARLY_STOP_PATIENCE = 5
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 63: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 64: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 65: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 66: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 67: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 68: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 69: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 70: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')


	# patience = 15
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 1000
	cf.EARLY_STOP_PATIENCE = 15
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 71: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 72: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 73: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 74: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 75: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 76: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 77: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 78: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')


	# patience = 30
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 1000
	cf.EARLY_STOP_PATIENCE = 30
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 79: ptraining_fn(nn.init_baseline(), experiment_num=args.exp, model_name='baseline')
	if args.exp == 80: ptraining_fn(nn.init_FCNN(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 81: ptraining_fn(nn.inti_lstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 82: ptraining_fn(nn.inti_gru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 83: ptraining_fn(nn.inti_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 84: ptraining_fn(nn.inti_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 85: ptraining_fn(nn.inti_cnn_bilstm(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 86: ptraining_fn(nn.inti_cnn_bigru(drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')

	# model small
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 87: ptraining_fn(nn.init_FCNN(layer_num = 3, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 88: ptraining_fn(nn.inti_lstm(layer_num=3, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 89: ptraining_fn(nn.inti_gru(layer_num=3, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 90: ptraining_fn(nn.inti_bilstm(bi_layer_num=3, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 91: ptraining_fn(nn.inti_bigru(bi_layer_num=3, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 92: ptraining_fn(nn.inti_cnn_bilstm(bi_layer_num=2, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 93: ptraining_fn(nn.inti_cnn_bigru(bi_layer_num=2, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')


	# model large
	cf.BATCH_SIZE = 64
	cf.EPOCHS = 50
	cf.EARLY_STOP_PATIENCE = None
	cf.LEARNING_RATE = 0.001 
	DROP_RATE = 0.3
	if args.exp == 94: ptraining_fn(nn.init_FCNN(layer_num = 10, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='FCNN')
	if args.exp == 95: ptraining_fn(nn.inti_lstm(layer_num=10, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='lstm')
	if args.exp == 96: ptraining_fn(nn.inti_gru(layer_num=10, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='gru')
	if args.exp == 97: ptraining_fn(nn.inti_bilstm(bi_layer_num=10, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bilstm')
	if args.exp == 98: ptraining_fn(nn.inti_bigru(bi_layer_num=10, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='bigru')
	if args.exp == 99: ptraining_fn(nn.inti_cnn_bilstm(bi_layer_num=8, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bilstm')
	if args.exp == 100: ptraining_fn(nn.inti_cnn_bigru(bi_layer_num=8, drop_rate=DROP_RATE), experiment_num=args.exp, model_name='cnn_bigru')





if __name__ == '__main__':
	parser = argparse.ArgumentParser("Exp Control")
	parser.add_argument("exp", help="", type=int, default=0)
	args = parser.parse_args()
	main(args)