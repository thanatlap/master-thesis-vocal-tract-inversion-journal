import numpy as np
import pandas as pd
from datetime import datetime
from time import time
from contextlib import redirect_stdout
import os
from os.path import join
import sys
import lib.dev_utils as utils
import scipy.stats

import model as nn
import config as cf

def log_performance(log, history):
	'''
	log performance of the model on training and validating subset.
	save result to log_dir/performace_.png
	'''
	history_graph_file = join('result','training', 'performance.png')
	utils.show_result(history, save_file = history_graph_file)
	log.write('Reference performance graph: %s\n'%history_graph_file)

def log_rmse_distribution(log,actual, pred, save_figure_dir, save_name, title,save_csv_dir, data_label=None):
	'''
	log rmse distribution both visualization and numerical result
	The rmse is computed using sklearn library
	'''
	# compute rmse
	test_rmse = utils.compute_rmse(actual,pred, axis=1)
	# get path
	rmse_dist_filename = join(save_figure_dir,save_name)
	# visulize and save distribution
	utils.rmse_distribution(test_rmse, rmse_dist_filename, title = title)
	log.write('Reference rmse distribution %s: %s\n'%(title,rmse_dist_filename))
	# create csv to store numerical result
	if data_label is not None:
		log.write('RMSE %s\n'%title)
		df = pd.DataFrame(data={'Label':np.load(data_label), 'Result': test_rmse})
		df.to_csv(join(save_csv_dir, 'dist_rmse_%s.csv'%title))
		log.write(df.to_string())

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

def _log_stat_dist(log,result, log_dir, measurement_name):
	'''
	log rmse distribution stat (Q1, Q2, Q3)
	The rmse is computed using sklearn library
	'''
	log_file = join(log_dir,'stat_result.csv')

	log.write('Stat of %s\n'%measurement_name)
	log.write('Min: %s\n'%np.percentile(result, 0))
	log.write('Q1: %s\n'%np.percentile(result, 25))
	log.write('Q2: %s\n'%np.percentile(result, 50))
	log.write('Q3: %s\n'%np.percentile(result, 75))
	log.write('Max: %s\n'%np.percentile(result, 100))

	log_stat = {'Metric':measurement_name,
	'Min':np.percentile(result, 0),
	'Q1':np.percentile(result, 25),
	'Q2':np.percentile(result, 50),
	'Q3':np.percentile(result, 75),
	'Max':np.percentile(result, 100),
	'IQR':np.percentile(result, 75)-np.percentile(result, 25),
	'AVG':np.mean(result),
	'Pop_SD':np.std(result),
	'SE':np.std(result,ddof=1)
	}

	if os.path.exists(log_file):
		log_df = pd.read_csv(log_file)
		log_df = log_df.append(log_stat, ignore_index=True)
	else:
		log_df = pd.DataFrame(log_stat, index=[-1])

	log_df.to_csv(log_file, index=False)


def log_stat_distribution(log,actual, pred, log_dir, subset):
	'''
	log rmse distribution stat (Q1, Q2, Q3)
	The rmse is computed using sklearn library
	'''
	test_rmse = utils.compute_rmse(actual,pred, axis=1)
	test_r2 = utils.compute_R2(actual,pred)
	_log_stat_dist(log, test_rmse, log_dir, measurement_name='rmse_%s'%subset)
	_log_stat_dist(log, test_r2, log_dir, measurement_name='r2_%s'%subset)

def log_rmse_parameter(log,actual, pred, save_csv_dir, title, scale=True):
	'''
	log rmse for each vocaltract point
	The rmse is computed using sklearn library
	'''
	# compute rmse by row (result rmse for each column)
	rmse_by_param = utils.compute_rmse(actual,pred, axis=0)
	log.write('Evaluation Result (RMSE)\nMean RMSE: %.3f\n'%np.mean(rmse_by_param))
	for idx, each_param_name in enumerate(utils.param_name):
		# if index is WC, print 'not predict'
		if idx == 8:
			log.write('WC RMSE: (Not predict)\n')
		else:
			if scale and (idx > 8):
				# the transform param doesnt have WC, thus, the label index is one step behind.
				log.write('%s RMSE: %.2f\n'%(each_param_name, rmse_by_param[idx-1]))
			else:
				# else, the inverse transform param already contain WC and can be skip
				log.write('%s RMSE: %.2f\n'%(each_param_name, rmse_by_param[idx]))
	# to have an equal size param, WC is removed if the label is transform
	if scale:
		label_name = utils.param_name.tolist().remove("WC")
	else:
		label_name =utils.param_name
	# log to csv
	pd.DataFrame(data={'Label':label_name, 'RMSE': rmse_by_param}).to_csv(join(save_csv_dir,'rmse_%s.csv'%title))

# log r2 of each parameter
def log_r2(log,actual, pred, save_csv_dir, title, scale=True):
	R2_by_param = utils.compute_R2(actual,pred)
	log.write('Evaluation Result R2\nMean R-Squared: %.4f\n'%np.mean(R2_by_param))
	for idx, each_param_name in enumerate(utils.param_name):
		if idx == 8:
			log.write('WC R2: (Not predict)\n')
		else:
			if scale and (idx > 8):
				# exclude WC when scale is True
				log.write('%s R2: %.4f\n'%(each_param_name, R2_by_param[idx-1]))
			else:
				log.write('%s R2: %.4f\n'%(each_param_name, R2_by_param[idx]))
	if scale:
		label_name = utils.param_name.tolist().remove("WC")
	else:
		label_name =utils.param_name
	pd.DataFrame(data={'Label':label_name, 'R2': R2_by_param}).to_csv(join(save_csv_dir,'r2_%s.csv'%title))

# log cc of each parameter
def log_cc(log,actual, pred, save_csv_dir, title, scale=True):
	corr = utils.compute_pearson_correlation(actual,pred)
	log.write('Evaluation Result (CC)\nMean Pearson Correlation Coef: %.3f\n'%np.mean(corr))
	for idx, each_param_name in enumerate(utils.param_name):
		if idx == 8:
			log.write('WC CC: (Not predict)\n')
		else:
			if scale and (idx > 8):
				log.write('%s CC: %.2f\n'%(each_param_name, corr[idx-1]))
			else:
				log.write('%s CC: %.2f\n'%(each_param_name, corr[idx]))
	if scale:
		label_name = utils.param_name.tolist().remove("WC")
	else:
		label_name =utils.param_name
	pd.DataFrame(data={'Label':label_name, 'CC': corr}).to_csv(join(save_csv_dir,'cc_%s.csv'%title))

# log formant
def log_formant(log, target_sound, target_label, estimated_sound, formant_dir):
	F1_re, F2_re, F3_re = utils.compute_formant_relative_error(target_sound, estimated_sound, formant_dir, cf.PRAAT_EXE, cf.DI_SYLLABLE)
	if log is not None:
		log.write('Relative error of formant\n')
		log.write('F1 mean relative error: %s\n'%np.mean(F1_re))
		log.write('F2 mean relative error: %s\n'%np.mean(F2_re))
		log.write('F3 mean relative error: %s\n'%np.mean(F3_re))
	# debug
	print(F1_re.shape)
	print(F2_re.shape)
	print(F3_re.shape)
	print(target_label.shape)
	formant_df = pd.DataFrame(data={'Label':target_label, 'F1': F1_re, 'F2': F2_re, 'F3': F3_re})
	formant_df.to_csv(join(formant_dir,'formant_df.csv'), index=False)
	log.write(formant_df.to_string())

# plot formant plot
def log_format_plot(log, formant_dir, save_csv_dir, label_name_path):
	label_name = np.load(label_name_path)
	sel_point = 20 if cf.DI_SYLLABLE else 13
	utils.export_format_csv(label_name, formant_dir ,sel_point = sel_point)

# Add to main log file
def log_experiment_csv_train(experiment_num, X_train, y_train, result, r2, training_result, training_r2, total_time):

	os.makedirs('result', exist_ok=True)
	log_file = join('..','log_experiment_n2.csv')

	load_save = cf.LOAD_FROM_SAVE if cf.LOAD_FROM_SAVE != None else np.NaN
	load_checkpoint = cf.LOAD_FROM_CHECKPOINT if cf.LOAD_FROM_CHECKPOINT != None else np.NaN

	log_exp = {'Experiment_no':experiment_num,
				'Disyllable':str(cf.DI_SYLLABLE),
				'Date':datetime.now().strftime("%Y-%B-%d %H:%M"),
				'Dataset': cf.DATASET_DIR[5:],
				'Model':cf.MODEL_DETAIL,
				'Train Loss': training_result[0],
				'Train RMSE': training_result[1],
				'Train R2': training_r2,
				'Test Loss':result[0],
				'Test RMSE':result[1],
				'Test R2':r2,
				'Eval RMSE':np.NaN,
				'Eval R2':np.NaN,
				'Training_time': total_time,
				'Loss function': cf.LOSS_FN,
				'Batch size':cf.BATCH_SIZE,
				'Epochs': cf.EPOCHS,
				'Learning rate': cf.LEARNING_RATE,
				'Train feature shape':str(X_train.shape),
				'Label shape':str(y_train.shape),
				'test_size(%)':cf.TEST_SIZE,
				'val_size(%)':cf.VAL_SIZE,
				'Load_save':load_save,
				'Load_checkpoint':load_checkpoint}

	if os.path.exists(log_file):
		log_df = pd.read_csv(log_file)
		log_df = log_df.append(log_exp, ignore_index=True)
	else:
		log_df = pd.DataFrame(log_exp, index=[-1])

	log_df.to_csv(log_file, index=False)

# add eval result to main log
def log_experiment_csv_eval(experiment_num, result, r2):

	log_file = join('..','log_experiment_n2.csv')
	if os.path.exists(log_file):
		log_df = pd.read_csv(log_file)
		log_df.loc[log_df['Experiment_no'] == experiment_num, ['Eval RMSE']] = result[1]
		log_df.loc[log_df['Experiment_no'] == experiment_num, ['Eval R2']] = r2
		log_df.to_csv(log_file, index=False)
	else:
		print('Main log not found!')
		print('Need to create main log in training file first!')

def log_result_train(experiment_num, X_train, X_val, X_test, y_train, y_val, y_test,y_pred, result, r2, history, model, training_y_pred, training_result, training_r2, total_time):

	# create log directory
	log_dir = join('result','training')
	os.makedirs(log_dir, exist_ok=True)
	# create csv log directory
	log_csv_dir = join(log_dir,'csv')
	os.makedirs(log_csv_dir,exist_ok=True)
	# log experiment to csv file
	log_experiment_csv_train(experiment_num, X_train, y_train, result, r2, training_result, training_r2, total_time)

	log = utils.get_log()
	log.write('========================================================\n')
	log.write('EXPERIMENT DATETIME %s\n'%datetime.now().strftime("%Y %B %d %H:%M"))
	log.write('EXPERIMENT #%s\n'%experiment_num)
	if cf.DI_SYLLABLE:
		log.write('PROJECT-TYPE: Di-syllablic vowel\n')
	else:
		log.write('PROJECT-TYPE: Mono-syllablic vowel\n')
	log.write('DESCRIPTION:\n%s\n'%cf.EXP_DESCRIPTION)
	log.write('--------------------------------------------------------\n')
	log.write('REFERENCE DATASET: %s\n'%cf.DATASET_DIR)
	log.write('Percent validating subset size: %s\n'%cf.VAL_SIZE)
	log.write('Percent testing subset size: %s\n'%cf.TEST_SIZE)
	log.write('Normalize features (MFCCs)\n')
	log.write('Training set: %s %s\nValidatetion set: %s %s\nTesting set: %s %s\n'%(str(X_train.shape),str(y_train.shape),str(X_val.shape),str(y_val.shape),str(X_test.shape),str(y_test.shape)))
	log.write('--------------------------------------------------------\n')
	log.write('Load previous model: %s\n'%str(cf.LOAD_FROM_SAVE))
	log.write('Load model from checkpoints: %s\n'%str(cf.LOAD_FROM_CHECKPOINT))
	log.write('About model: %s\n'%cf.MODEL_DETAIL)
	with redirect_stdout(log):
		model.summary()
	log.write('\n')
	log.write('Optimizer: Adam\n')
	log.write('Use AMSGrad: %s\n'%str(cf.AMSGRAD))
	log.write('Learning rate: %.3f\n'%cf.LEARNING_RATE)
	log.write('Beta1: %.3f\n'%cf.BETA1)
	log.write('Beta2: %.3f\n'%cf.BETA1)
	log.write('Epsilon: %s\n'%str(cf.EPS))
	log.write('Schedule_decay: %.3f\n'%cf.SDECAY)
	log.write('Loss function: %s\n'%cf.LOSS_FN)
	log.write('Metric: RMSE\n')
	log.write('--------------------------------------------------------\n')
	log.write('Training configurations\n')
	log.write('Batch size: %s\n'%cf.BATCH_SIZE)
	log.write('Epochs: %s\n'%cf.EPOCHS)
	log.write('Checkpoint period: %s\n'%cf.CHECKPOINT_PEROID)
	log.write('--------------------------------------------------------\n')
	log.write('TEST RESULT\n')
	log.write('Loss: %.5f\nRMSE: %.5f\n'%(result[0],result[1]))
	log.write('Result R2: %.5f\n'%(r2))
	log.write('Training Time:  %.3fs'%total_time)
	log.write('--------------------------------------------------------\n')
	log_performance(log, history)
	log.write('Reference performance log: %s\n'%join(log_dir,'log_experiment_%s.csv'%experiment_num))
	# RMSE distribution of testing subset
	log_rmse_distribution(log, y_test, y_pred, 
		save_figure_dir= log_dir,  
		save_name='rmse_distribution_test_subset_model_%s.png'%experiment_num,
		title='test_subset_RMSE',
		save_csv_dir = log_csv_dir)
	log_stat_distribution(log, y_train, training_y_pred, log_dir, subset='training')
	log_stat_distribution(log, y_test, y_pred, log_dir, subset='testing')
	#Compare RMSE of each parameter
	log_rmse_parameter(log, y_test, y_pred, log_dir, title='scale_train')
	# calculate R2
	log_r2(log, y_test, y_pred, log_dir, title='scale_train')
	# log cc
	log_cc(log, y_test, y_pred, log_dir, title='scale_train')
	log.write('========================================================\n')
	log.close()

def log_result_eval(actual_label, y_pred, eval_result, r2, target_sound, estimated_sound, exp_num):

	# prepare path
	log_dir = join('result', 'eval') 
	os.makedirs(log_dir, exist_ok=True)
	# log to the main csv
	log_experiment_csv_eval(exp_num, eval_result, r2)
	# create csv log directory
	log_csv_dir = join(log_dir,'csv')
	os.makedirs(log_csv_dir,exist_ok=True)
	# get log
	log = utils.get_log()
	log.write('========================================================\n')
	log.write('DATETIME %s\n'%datetime.now().strftime("%Y %B %d %H:%M"))
	log.write('EXPERIMENT #%s\n'%exp_num)
	if cf.DI_SYLLABLE:
		log.write('PROJECT-TYPE: Di-syllablic vowel\n')
	else:
		log.write('PROJECT-TYPE: Mono-syllablic vowel\n')
	log.write('--------------------------------------------------------\n')
	log.write('Result Loss: %.4f\nRMSE: %.4f\n'%(eval_result[0],eval_result[1]))
	log.write('Result R2: %.4f\n'%(r2))
	log.write('--------------------------------------------------------\n')
	log.write('Evaluate with Transform Label\n')
	# get label name path
	label_name_path = join(cf.EVALSET_DIR,'syllable_name.npy')
	#RMSE of each vowel before scaling
	log_rmse_distribution(log, actual_label, y_pred, 
		save_figure_dir = log_dir, 
		save_name='rmse_eval_scale_%s.png'%exp_num, 
		title='Eval_Scale_Parameters',
		save_csv_dir=log_csv_dir,
		data_label = label_name_path)
	log_stat_distribution(log,actual_label, y_pred,log_dir, subset='eval')
	#Compare RMSE of each parameter
	log_rmse_parameter(log, actual_label, y_pred, log_csv_dir, title='scale_eval')
	log.write('\n')
	log.write('--------------------------------------------------------\n')
	log.write('Evaluate with Inverse-transform Label\n')
	t_actual_label = utils.label_transform(actual_label, invert=True)
	t_y_pred = utils.label_transform(y_pred, invert=True)
	#RMSE of each vowel after descaling
	log_rmse_distribution(log, 
		t_actual_label, 
		t_y_pred, 
		save_figure_dir=log_dir, 
		save_name='rmse_eval_descale_%s.png'%exp_num, 
		title='Eval_Descale_Parameters',
		save_csv_dir=log_csv_dir,
		data_label = label_name_path)
	log_stat_distribution(log,t_actual_label, t_y_pred, log_dir, subset='trans_eval')
	#Compare RMSE of each parameter
	log_rmse_parameter(log, t_actual_label, t_y_pred, log_csv_dir, title='descale_eval',scale=False)
	log.write('\n')
	log.write('--------------------------------------------------------\n')
	# CC
	# create csv log directory
	log_scatter = join(log_dir,'scatter')
	os.makedirs(log_csv_dir,exist_ok=True)
	# log CC
	log_cc(log, t_actual_label, t_y_pred,log_csv_dir, title='descale',scale=False)
	utils.plot_scatter(t_actual_label,
		t_y_pred, 
		log_scatter)
	log.write('Scale Label R2\n')
	log_r2(log,actual_label, y_pred, log_csv_dir, title='scale')
	# calculate R2
	log.write('Descale Label R2\n')
	log_r2(log,t_actual_label, t_y_pred,log_csv_dir, title='descale', scale=False)
	# calculate formant
	log.write('\n')
	log.write('---------------------------------\n')
	# create csv log directory
	formant_dir = join(log_dir,'formant')
	os.makedirs(formant_dir,exist_ok=True)
	log_formant(log, target_sound, np.load(label_name_path), estimated_sound, formant_dir)
	log_format_plot(log,formant_dir, log_dir, label_name_path)
	log.write('--------------------------------------------------------\n')
	log.write('Reference directory\n')
	log.write('Log directory: %s\n'%log_dir)
	log.write('========================================================\n')
	log.close()

def log_result_predict(y_pred, model_file, data_dir, output_dir, target_sound, predict_sound, syllable_name):
	log_dir = join('result', 'predict')
	os.makedirs(log_dir, exist_ok=True)
	log = utils.get_log()
	log.write('========================================================\n')
	log.write('DATETIME %s\n'%datetime.now().strftime("%Y %B %d %H:%M"))
	if cf.DI_SYLLABLE:
		log.write('PROJECT-TYPE: Di-syllablic vowel\n')
	else:
		log.write('PROJECT-TYPE: Mono-syllablic vowel\n')
	# calculate formant
	log.write('---------------------------------\n')
	formant_dir = join(log_dir,'formant')
	os.makedirs(formant_dir,exist_ok=True)
	log_formant(log, target_sound, syllable_name, predict_sound, formant_dir)
	log.write('--------------------------------------------------------\n')
	log.write('Reference directory\n')
	log.write('Log directory: %s\n'%log_dir)
	log.write('Model directory: %s\n'%model_file)
	log.write('\n')
	log.write('========================================================\n')