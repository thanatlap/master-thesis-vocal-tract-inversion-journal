from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
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
import model_2 as nn
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

	callback_list = []

	if cf.EARLY_STOP_PATIENCE:
		# Early stop
		early = callbacks.EarlyStopping(monitor='val_R2', 
			min_delta=0, patience=cf.EARLY_STOP_PATIENCE, 
			verbose=1, mode='max', baseline=None, restore_best_weights=cf.SAVE_BEST_WEIGHT)
		callback_list.extend([early])

	if cf.CHECKPOINT_PEROID:
		# Checkpoint
		checkpoint = callbacks.ModelCheckpoint(filepath=join('model','checkpoint','weights.{epoch:02d}-{val_rmse:.4f}.h5'), 
												monitor='val_rmse', verbose=1, mode='min',save_best_only=True, period = cf.CHECKPOINT_PEROID)
		callback_list.extend([checkpoint])

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

def training_fn(model_fn, X_train, X_val, X_test, y_train, y_val, y_test, experiment_num, model_name='undefined'):
	'''
	warpper function to training on each experiment
	This function is created for training multiple model in one run.
	'''
	print('TRAINING EXPERIMENT#: {}'.format(experiment_num))
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

def main(args):
	
	print('[DEBUG] Time: {}'.format(datetime.now()))
	print('[DEBUG] Data: {}'.format(cf.DATASET_DIR))

	if args.exp in range(0,7) or args.exp in range(33,39) or args.exp in range(70,80):
		cf.DATASET_DIR = '../data/m_dataset_p2/prep_data_13'
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

	if args.exp == 0: ptraining_fn(nn.init_baseline(), model_name='baseline')
	if args.exp == 1: ptraining_fn(nn.init_FCNN(), model_name='FCNN')
	if args.exp == 2: ptraining_fn(nn.init_bilstm(), model_name='bilstm')
	if args.exp == 3: ptraining_fn(nn.init_LTRCNN(), model_name='LTRCNN')
	if args.exp == 4: ptraining_fn(nn.init_senet(embedded_path = None),model_name='senet')
	if args.exp == 5: ptraining_fn(nn.init_senet(se_enable=False, embedded_path = None),model_name='resnet')
	if args.exp == 6: ptraining_fn(nn.init_senet(embedded_path = 'model/between_embedded_32.hdf5'), model_name='senet_em')
	if args.exp == 33: ptraining_fn(nn.init_lstm(), model_name='lstm')
	if args.exp == 34: ptraining_fn(nn.init_cnn_bilstm(embedded_path = None), model_name='cnn_bilstm')
	if args.exp == 35: ptraining_fn(nn.init_bilstm(bi_layer_num=4), model_name='bilstm')
	if args.exp == 36: ptraining_fn(nn.init_lstm(unit=256), model_name='lstm')
	if args.exp == 37: ptraining_fn(nn.init_bilstm(), model_name='bilstm')
	if args.exp == 38: ptraining_fn(nn.init_lstm(), model_name='lstm')
	if args.exp==71: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64), model_name='conv_bilistm' )
	if args.exp==72: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64, embedded_path = 'model/between_embedded_32.hdf5'), model_name='conv_bilistm' )

	if args.exp in [52, 53]:
		cf.EARLY_STOP_PATIENCE = 7

	if args.exp in range(7,14) or args.exp in range(20,33) or args.exp in range(39,70):
		cf.DATASET_DIR = '../data/d_dataset_3/prep_data_13'
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

	if args.exp == 7: ptraining_fn(nn.init_baseline(), model_name='baseline')
	if args.exp == 8: ptraining_fn(nn.init_FCNN(), model_name='FCNN')
	if args.exp == 9: ptraining_fn(nn.init_bilstm(), model_name='bilstm')
	if args.exp == 10: ptraining_fn(nn.init_LTRCNN(), model_name='LTRCNN')
	if args.exp == 23: ptraining_fn(nn.init_senet(embedded_path = None), model_name='senet')
	if args.exp == 24: ptraining_fn(nn.init_senet(se_enable=False, embedded_path = None),model_name='resnet')
	if args.exp == 13: ptraining_fn(nn.init_senet(embedded_path = 'model/between_embedded_32.hdf5'), model_name='senet_em')
	if args.exp == 25: ptraining_fn(nn.init_senet(bilstm=3, bilstm_unit=256, embedded_path = None), model_name='senet')

	if args.exp==26: ptraining_fn(nn.init_senet(xx=False), model_name='senet' )
	if args.exp == 27: ptraining_fn(nn.init_cnn_bilstm(embedded_path = None), model_name='cnn_bilstm')
	if args.exp == 28: ptraining_fn(nn.init_bilstm(output_act='tanh'), model_name='bilstm')
	if args.exp == 29: ptraining_fn(nn.init_bilstm(bi_layer_num=4), model_name='bilstm')
	if args.exp == 30: ptraining_fn(nn.init_lstm(), model_name='lstm')
	if args.exp == 31: ptraining_fn(nn.init_bilstm(), model_name='bilstm')
	if args.exp == 32: ptraining_fn(nn.init_lstm(unit=256), model_name='lstm')
	if args.exp == 39: ptraining_fn(nn.init_cnn_bilstm(output_act='tanh', embedded_path = None), model_name='cnn_bilstm')
	if args.exp==40: ptraining_fn(nn.init_senet(se_enable=False, output_act='tanh'), model_name='resnet' )
	if args.exp==41: ptraining_fn(nn.init_conv_bilistm(), model_name='conv_bilistm' )
	if args.exp==43: ptraining_fn(nn.init_conv_bilistm_2(), model_name='conv_bilistm' )

	# try different model
	if args.exp==44: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64), model_name='conv_bilistm' )
	if args.exp==45: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64, reg=0.0005), model_name='conv_bilistm' )
	if args.exp==46: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64, embedded_path = 'model/between_embedded_32.hdf5'), model_name='conv_bilistm' )
	if args.exp==47: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=128, embedded_path = 'model/between_embedded_32.hdf5'), model_name='conv_bilistm' )
	if args.exp==48: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64, bilstm=4), model_name='conv_bilistm' )
	if args.exp==49: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64, bilstm=4, reg=0.0005), model_name='conv_bilistm' )
	if args.exp==50: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64, bilstm=4, embedded_path = 'model/between_embedded_32.hdf5'), model_name='conv_bilistm' )
	if args.exp==51: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=128, bilstm=4, reg=0.0005, embedded_path = 'model/between_embedded_32.hdf5'), model_name='conv_bilistm' )
	if args.exp==52: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64), model_name='conv_bilistm' )
	if args.exp==53: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64, reg=0.0005), model_name='conv_bilistm' )
	if args.exp==54: ptraining_fn(nn.init_conv_bilistm(cnn_unit=64), model_name='conv_bilistm' )
	if args.exp == 55: ptraining_fn(nn.init_senet(feature_layer=2, bilstm=5, embedded_path = None), model_name='senet')
	if args.exp==56: ptraining_fn(nn.init_conv_bilistm_2(cnn_unit=64), model_name='conv_bilistm' )
	if args.exp == 27: ptraining_fn(nn.init_cnn_bilstm(embedded_path = None), model_name='cnn_bilstm')

	if args.exp == 14: 
		cf.DATASET_DIR = '../data/d_nospeaker_1/prep_data_13'
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
		training_fn(nn.init_senet(embedded_path = None),
							X_train=X_train, 
							X_val=X_val, 
							X_test=X_test, 
							y_train=y_train, 
							y_val=y_val, 
							y_test=y_test,
							experiment_num=args.exp, 
							model_name='senet')

	if args.exp == 15: 
		cf.DATASET_DIR = '../data/d_nospeaker_1/prep_data_13_noaug'
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
		training_fn(nn.init_senet(embedded_path = None),
							X_train=X_train, 
							X_val=X_val, 
							X_test=X_test, 
							y_train=y_train, 
							y_val=y_val, 
							y_test=y_test,
							experiment_num=args.exp, 
							model_name='senet')

	if args.exp == 16: 
		cf.DATASET_DIR = '../data/d_dataset_3/prep_data_13_noaug'
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
		training_fn(nn.init_senet(embedded_path = None),
							X_train=X_train, 
							X_val=X_val, 
							X_test=X_test, 
							y_train=y_train, 
							y_val=y_val, 
							y_test=y_test,
							experiment_num=args.exp, 
							model_name='senet')

	if args.exp == 17: 
		cf.DATASET_DIR = '../data/d_nospeaker_1/prep_data_13'
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
		training_fn(nn.init_senet(embedded_path = 'model/between_embedded_32.hdf5'),
							X_train=X_train, 
							X_val=X_val, 
							X_test=X_test, 
							y_train=y_train, 
							y_val=y_val, 
							y_test=y_test,
							experiment_num=args.exp, 
							model_name='senet_em')

	if args.exp == 18: 
		cf.DATASET_DIR = '../data/d_nospeaker_1/prep_data_13_noaug'
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
		training_fn(nn.init_senet(embedded_path = 'model/between_embedded_32.hdf5'),
							X_train=X_train, 
							X_val=X_val, 
							X_test=X_test, 
							y_train=y_train, 
							y_val=y_val, 
							y_test=y_test,
							experiment_num=args.exp, 
							model_name='senet_em')

	if args.exp == 19: 
		cf.DATASET_DIR = '../data/d_dataset_3/prep_data_13_noaug'
		X_train, X_val, X_test, y_train, y_val, y_test = prep_data()
		training_fn(nn.init_senet(embedded_path = 'model/between_embedded_32.hdf5'),
							X_train=X_train, 
							X_val=X_val, 
							X_test=X_test, 
							y_train=y_train, 
							y_val=y_val, 
							y_test=y_test,
							experiment_num=args.exp, 
							model_name='senet_em')
	

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Exp Control")
	parser.add_argument("exp", help="", type=int, default=0)
	args = parser.parse_args()
	main(args)