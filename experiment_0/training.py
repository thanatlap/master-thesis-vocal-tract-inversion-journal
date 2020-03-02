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
		print('[INFO] Load from save: %s'%cf.LOAD_FROM_SAVE)
		return tf.keras.models.load_model(join('model',cf.LOAD_FROM_SAVE), custom_objects={'rmse': nn.rmse, 'R2':nn.R2})
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
		elif cf.OPT_NUM == 4:
			lr_schedule = optimizers.schedules.ExponentialDecay(
				cf.LEARNING_RATE,
				decay_steps=100000,
				decay_rate=0.96,
				staircase=True)
			opt = optimizers.SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True, name='SGD')

		
		model.compile(optimizer=opt,loss=cf.LOSS_FN,metrics=[nn.rmse, nn.R2])
	return model

def training(features, labels, val_features, val_labels, model, model_name=None, experiment_num=None):
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
		early = callbacks.EarlyStopping(monitor='val_R2', 
			min_delta=0, patience=cf.EARLY_STOP_PATIENCE, 
			verbose=1, mode='max', baseline=None, restore_best_weights=False)
		
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
		batch_size=cf.BATCH_SIZE,
		epochs=cf.EPOCHS,
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
	loss = model.evaluate(features,labels,verbose=False)[0]
	y_pred = model.predict(features)
	r2 = utils.compute_R2(labels, y_pred,multioutput='uniform_average')
	rmse = utils.total_rmse(labels, y_pred)
	results = [loss, rmse, r2]
	return y_pred, results

def training_fn(model_fn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num=None, model_name=None):
	'''
	warpper function to training on each experiment
	This function is created for training multiple model in one run.
	'''
	# load experiment number
	# try:
	experiment_num = utils.get_experiment_number() if experiment_num is None else experiment_num
	print('### --- Training experiment number: %s'%experiment_num)
	# initialize/load model
	model = get_model(model_fn = model_fn, input_shape = (X_train.shape[1],X_train.shape[2]))
	# training
	history, total_time, early = training(X_train, y_train, X_val, y_val, model, model_name=model_name, experiment_num=experiment_num)
	training_y_pred, train_results = testing(X_train, y_train, model)
	val_y_pred, val_results = testing(X_val, y_val, model)
	y_pred, test_results = testing(X_test, y_test, model)
	print('\n\n[INFO] Result experiment number: #%s'%experiment_num)
	print(" -- |TRAINING  |  Loss: %.3f RMSE: %.3f R2:%.3f" %(train_results[0],train_results[1],train_results[2]))
	print(" -- |VALIDATING|  Loss: %.3f RMSE: %.3f R2:%.3f" %(val_results[0],val_results[1],val_results[2]))
	print(" -- |TESTING   |  Loss: %.3f RMSE: %.3f R2:%.3f\n\n" %(test_results[0],test_results[1],test_results[2]))
	print('[INFO] Log Results')
	# log experiment	
	res.log_result_train(experiment_num, X_train, X_val, X_test, y_train, y_val, y_test, 
		y_pred, test_results, history, model, 
		training_y_pred, train_results, total_time, model_name, early, 
		val_y_pred, val_results)
		
	# except Exception as e:
	# 	print('[ERROR] EXPERIMENT %d FAIL!\n%s'%(experiment_num,e))

def main(args):
	
	tf.random.set_seed(42)

	print('%s'%str(datetime.now()))

	X_train, X_val, X_test, y_train, y_val, y_test = prep_data()

	ptraining_fn = partial(training_fn, 
		X_train=X_train, 
		X_val=X_val, 
		X_test=X_test, 
		y_train=y_train, 
		y_val=y_val, 
		y_test=y_test,
		experiment_num=args.exp, 
		model_name='undefined')

	if args.exp == 43: ptraining_fn(nn.init_senet(),
		model_name='senet')
	if args.exp == 44: ptraining_fn(nn.init_senet(bilstm_unit=256),
		model_name='senet')
	if args.exp == 45: ptraining_fn(nn.init_senet(bilstm = 1, bilstm_unit=256),
		model_name='senet')
	if args.exp == 48: ptraining_fn(nn.init_senet(bilstm_unit=256),
		model_name='senet')
	if args.exp == 5: ptraining_fn(nn.init_baseline(), 
		model_name='baseline')
	if args.exp == 42: ptraining_fn(nn.init_senet_skip(), 
		model_name='senet_skip')
	if args.exp == 14: ptraining_fn(nn.init_LTRCNN(drop_rate=0.3), 
		model_name='LTRCNN')
	
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Exp Control")
	parser.add_argument("exp", help="", type=int, default=0)
	args = parser.parse_args()
	main(args)