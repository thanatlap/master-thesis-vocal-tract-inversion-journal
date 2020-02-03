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
		print('Result experiment number: #%s'%experiment_num)
		print('Result RMSE: %.4f'%(result[1]))
		print('Result R2: %.4f\n'%(r2))
		print('[INFO] Log Results')
		# log experiment	
		res.log_result_train(experiment_num, X_train, X_val, X_test, y_train, y_val, y_test,y_pred, result, r2, history, model, 
			training_y_pred, training_result, training_r2, total_time, model_name, early)
		
	except Exception as e:
		print('experiment %f fail'%experiment_num)
		print(e)

def main(args):

	if args.exp == 47:
		cf.DATASET_DIR = '../data/d_dataset_6/prep_exp10'
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()

	if args.exp == 48:
		cf.DATASET_DIR = '../data/d_dataset_7/prep_exp10'
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


	# baseline
	if args.exp == 0: ptraining_fn(nn.inti_bilstm(), experiment_num=0, model_name='inti_bilstm')

	# test learning rate
	if args.exp == 1: 
		cf.LEARNING_RATE = 0.01
		ptraining_fn(nn.inti_bilstm(), experiment_num=1, model_name='inti_bilstm')

	if args.exp == 2: 
		cf.LEARNING_RATE = 0.005
		ptraining_fn(nn.inti_bilstm(), experiment_num=2, model_name='inti_bilstm')

	if args.exp == 3: 
		cf.LEARNING_RATE = 0.0005
		ptraining_fn(nn.inti_bilstm(), experiment_num=3, model_name='inti_bilstm')

	if args.exp == 4: 
		cf.LEARNING_RATE = 0.0001
		ptraining_fn(nn.inti_bilstm(), experiment_num=4, model_name='inti_bilstm')

	if args.exp == 5: 
		cf.LEARNING_RATE = 0.00005
		ptraining_fn(nn.inti_bilstm(), experiment_num=5, model_name='inti_bilstm')

	if args.exp == 6: 
		cf.LEARNING_RATE = 0.1
		ptraining_fn(nn.inti_bilstm(), experiment_num=6, model_name='inti_bilstm')


	# test batch size

	if args.exp == 7: 
		cf.BATCH_SIZE = 32
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=5),
			experiment_num=7, model_name='inti_bilstm')

	if args.exp == 8: 
		cf.BATCH_SIZE = 64
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=5),
			experiment_num=8, model_name='inti_bilstm')

	if args.exp == 9: 
		cf.BATCH_SIZE = 256
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=5),
			experiment_num=9, model_name='inti_bilstm')

	# test earlystop

	if args.exp == 26: 
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=5),
			experiment_num=26, model_name='inti_bilstm')

	if args.exp == 27: 
		cf.EARLY_STOP_PATIENCE = 25
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=5),
			experiment_num=27, model_name='inti_bilstm')

	if args.exp == 28: 
		cf.EARLY_STOP_PATIENCE = 50
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=5),
			experiment_num=28, model_name='inti_bilstm')

	# test network

	if args.exp == 10: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=3),
			experiment_num=10, model_name='inti_bilstm')

	if args.exp == 11: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=4),
			experiment_num=11, model_name='inti_bilstm')

	if args.exp == 12: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=6),
			experiment_num=12, model_name='inti_bilstm')

	if args.exp == 13: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=7),
			experiment_num=13, model_name='inti_bilstm')

	if args.exp == 14: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=8),
			experiment_num=14, model_name='inti_bilstm')

	if args.exp == 15: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.5, bi_layer_num=9),
			experiment_num=15, model_name='inti_bilstm')

	# test dropout

	if args.exp == 16: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.4, bi_layer_num=5),
			experiment_num=16, model_name='inti_bilstm')

	if args.exp == 17: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.3, bi_layer_num=5),
			experiment_num=17, model_name='inti_bilstm')

	if args.exp == 18: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.2, bi_layer_num=5),
			experiment_num=18, model_name='inti_bilstm')

	if args.exp == 19: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=128, dropout_rate=0.1, bi_layer_num=5),
			experiment_num=19, model_name='inti_bilstm')

	# test capacity

	if args.exp == 20: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=64, dropout_rate=0.3, bi_layer_num=3),
			experiment_num=20, model_name='inti_bilstm')

	if args.exp == 21: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=64, dropout_rate=0.3, bi_layer_num=5),
			experiment_num=21, model_name='inti_bilstm')

	if args.exp == 22: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=64, dropout_rate=0.5, bi_layer_num=8),
			experiment_num=22, model_name='inti_bilstm')

	if args.exp == 23: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=256, dropout_rate=0.5, bi_layer_num=3),
			experiment_num=23, model_name='inti_bilstm')

	if args.exp == 24: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=256, dropout_rate=0.5, bi_layer_num=5),
			experiment_num=24, model_name='inti_bilstm')

	if args.exp == 25: 
		ptraining_fn(nn.inti_bilstm( unit_lstm=256, dropout_rate=0.5, bi_layer_num=7),
			experiment_num=25, model_name='inti_bilstm')

	if args.exp == 29: 
		ptraining_fn(nn.inti_cnn_bilstm(),
			experiment_num=29, model_name='cnn_bilstm')

	if args.exp == 30: 
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_bilstm(),
			experiment_num=30, model_name='cnn_bilstm')

	if args.exp == 31: 
		cf.EARLY_STOP_PATIENCE = 10
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=4),
			experiment_num=31, model_name='cnn_bilstm')

	if args.exp == 32: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 10
		ptraining_fn(nn.inti_cnn_bilstm(),
			experiment_num=32, model_name='cnn_bilstm')

	if args.exp == 33: 
		ptraining_fn(nn.inti_cnn_fc(),
			experiment_num=33, model_name='cnn_fc')

	if args.exp == 34: 
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_fc(),
			experiment_num=34, model_name='cnn_fc')

	if args.exp == 35: 
		cf.EARLY_STOP_PATIENCE = 10
		ptraining_fn(nn.inti_cnn_fc(cnn_layer=8),
			experiment_num=35, model_name='cnn_fc')

	if args.exp == 36: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 10
		ptraining_fn(nn.inti_cnn_fc(),
			experiment_num=36, model_name='cnn_fc')

	if args.exp == 37: 
		cf.EARLY_STOP_PATIENCE = 10
		ptraining_fn(nn.inti_cnn_fc(cnn_layer=3),
			experiment_num=37, model_name='cnn_fc')

	if args.exp == 38: 
		cf.EARLY_STOP_PATIENCE = 10
		ptraining_fn(nn.inti_cnn_fc(dense_layer_num=2),
			experiment_num=38, model_name='cnn_fc')

	if args.exp == 39: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=5, unit_lstm=64, dropout_rate=0.3, bi_layer_num=3),
			experiment_num=39, model_name='cnn_bilstm')

	if args.exp == 40: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=5, unit_lstm=64, dropout_rate=0.3),
			experiment_num=40, model_name='cnn_bilstm')

	if args.exp == 41: 
		cf.LEARNING_RATE = 0.00001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=5, unit_lstm=64, dropout_rate=0.3),
			experiment_num=41, model_name='cnn_bilstm')

	if args.exp == 42: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_bilstm( unit_lstm=64, dropout_rate=0.3, bi_layer_num=3),
			experiment_num=42, model_name='inti_bilstm')

	if args.exp == 43: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_bilstm(), experiment_num=43, model_name='inti_bilstm')

	if args.exp == 44: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		cf.LOSS_FN = [nn.huber_loss]
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=5, unit_lstm=64, dropout_rate=0.3, bi_layer_num=3),
			experiment_num=44, model_name='cnn_bilstm')

	if args.exp == 45: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=5, unit_lstm=64, dropout_rate=0.3, bi_layer_num=3),
			experiment_num=45, model_name='cnn_bilstm')

	if args.exp == 46: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=5, unit_lstm=64, dropout_rate=0.3, bi_layer_num=3),
			experiment_num=46, model_name='cnn_bilstm')

	if args.exp == 47: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=5, unit_lstm=64, dropout_rate=0.3, bi_layer_num=3),
			experiment_num=47, model_name='cnn_bilstm')

	if args.exp == 48: 
		cf.LEARNING_RATE = 0.0001
		cf.EARLY_STOP_PATIENCE = 5
		ptraining_fn(nn.inti_cnn_bilstm(cnn_layer=5, unit_lstm=64, dropout_rate=0.3, bi_layer_num=3),
			experiment_num=48, model_name='cnn_bilstm')


if __name__ == '__main__':
	parser = argparse.ArgumentParser("Exp Control")
	parser.add_argument("exp", help="", type=int, default=0)
	args = parser.parse_args()
	main(args)