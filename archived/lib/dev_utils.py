import numpy as np
import pandas as pd
import os
from os.path import join
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
import librosa
from librosa import feature
from scipy.stats.stats import pearsonr   
from sklearn.metrics import r2_score
import itertools
import math
import praatio
from praatio import praat_scripts
from praatio.utilities import utils as praat_utils
import warnings
import argparse
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.ticker import MaxNLocator

param_high = np.array([1, -3.5, 0, 0, 1, 4, 1, 1, 1, 4, 1, 5.5, 2.5, 4, 5, 2, 0, 1.4, 1.4, 1.4, 1.4, 0.3, 0.3, 0.3])
param_low = np.array([0,-6.0, -0.5, -7.0, -1.0, -2.0, 0, -0.1, 0, -3, -3, 1.5, -3.0, -3, -3, -4, -6, -1.4, -1.4, -1.4, -1.4, -0.05, -0.05, -0.05]) 
param_name = np.array(["HX","HY","JX","JA","LP","LD","VS","VO","WC","TCX","TCY","TTX","TTY","TBX","TBY","TRX","TRY","TS1","TS2","TS3","TS4","MA1","MA2","MA3"])

DEL_PARAMS_LIST = [2,8,15,16,21,22,23]

def str2bool(v):
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

#Function for preprocessing data
def delete_params(params):
	'''
	This function remove JX, WC, TRX, TRY, and MS1,2,3 paramter
	'''
	return np.delete(params,DEL_PARAMS_LIST , axis=1)

# def normalize_mfcc_by_mean_cepstral(mfcc):
# 	ceps_mean = np.mean([np.mean(item, axis=0) for item in mfcc], axis=0)
# 	ceps_std = np.mean([np.std(item, axis=0) for item in mfcc], axis=0)
# 	return np.array([(item-ceps_mean)/(ceps_std+1e-8) for item in mfcc])

def normalize_mfcc_by_mean_cepstral(mfcc, is_train, is_disyllable, mode, X_shape):
	vars_dir = 'vars'
	filename = 'feature_scaler_di.npy' if is_disyllable else 'feature_scaler_mono.npy' 
	filepath = join(vars_dir, filename)
	os.makedirs(vars_dir, exist_ok=True)


	mfcc = np.array(mfcc)
	if is_train:
		# compute mean and variance excluding augmented data
		if X_shape:
			original_feat = mfcc[:int(X_shape*2)] if is_disyllable else mfcc[:int(X_shape)]
		else:
			original_feat = mfcc

		original_feat = np.array(original_feat)
		print(original_feat.shape)

		ceps_mean = np.mean([np.mean(item, axis=0) for item in original_feat], axis=0)
		ceps_std = np.mean([np.std(item, axis=0) for item in original_feat], axis=0)

		np.save(filepath,np.array([ceps_mean, ceps_std]))
	else:
		if mode == 'predict' and mfcc.shape[0] > 79:
			prep_file = join(vars_dir, 'prep_pred.npy')
			if os.path.isfile(prep_file):
				mean_std = np.load(prep_file).tolist()
				ceps_mean = np.array(mean_std[0])
				ceps_std = np.array(mean_std[1])
			else:
				ceps_mean = np.mean([np.mean(item, axis=0) for item in mfcc], axis=0)
				ceps_std = np.mean([np.std(item, axis=0) for item in mfcc], axis=0)
				np.save(prep_file,np.array([ceps_mean, ceps_std]))

			# ceps_mean = np.mean([np.mean(item, axis=0) for item in mfcc], axis=0)
			# ceps_std = np.mean([np.std(item, axis=0) for item in mfcc], axis=0)
		else:
			if os.path.isfile(filepath):
				mean_std = np.load(filepath).tolist()
				ceps_mean = np.array(mean_std[0])
				ceps_std = np.array(mean_std[1])
			else:
				raise ValueError('File %s doest exist'%vars_dir)
		# if os.path.isfile(filepath):
		# 		mean_std = np.load(filepath).tolist()
		# 		ceps_mean = np.array(mean_std[0])
		# 		ceps_std = np.array(mean_std[1])
		# else:
		# 	raise ValueError('File %s doest exist'%vars_dir)

		# mean_std = np.load(filepath).tolist()
		# ceps_mean = np.array(mean_std[0])
		# ceps_std = np.array(mean_std[1])

		
		# mean_std = np.load(filepath).tolist()
		# ceps_mean_1 = np.array(mean_std[0])
		# ceps_std_1 = np.array(mean_std[1])

		# prep_file = join(vars_dir, 'prep_pred.npy')
		# mean_std = np.load(prep_file).tolist()
		# ceps_mean_2 = (np.array(mean_std[0])+ceps_mean_1)/2
		# ceps_std_2 = (np.array(mean_std[1])+ceps_std_1)/2

		# ceps_mean = (np.mean([np.mean(item, axis=0) for item in mfcc], axis=0) + ceps_mean_1)/2
		# ceps_std = (np.mean([np.std(item, axis=0) for item in mfcc], axis=0) + ceps_std_1)/2

		ceps_mean = np.mean([np.mean(item, axis=0) for item in mfcc], axis=0)
		ceps_std = np.mean([np.std(item, axis=0) for item in mfcc], axis=0)

		# ceps_mean = (ceps_mean+ceps_mean_2)/2
		# ceps_std = (ceps_std+ceps_std_2)/2

	return np.array([(item-ceps_mean)/(ceps_std+1e-8) for item in mfcc])

def standardize_mfcc(features, is_train, is_disyllable, self_centering=False):

	if not self_centering:
		vars_dir = 'vars'
		filename = 'mean_std_di.npy' if is_disyllable else 'mean_std_mono.npy' 
		os.makedirs(vars_dir, exist_ok=True)
		
		# find mean of each feature in each timestep
		if is_train:
			mean = np.mean(features, axis=0)
			std = np.std(features, axis=0)
			np.save(os.path.join(vars_dir, filename),np.array([mean, std]))
		else:
			if os.path.isfile(os.path.join(vars_dir, filename)):
				mean_std = np.load(os.path.join(vars_dir, filename)).tolist()
				mean = np.array(mean_std[0])
				std = np.array(mean_std[1])
			else:
				raise ValueError('File %s doest exist'%vars_dir)
		features = (features - mean)/(std+1e-8)

	else:
		# self centering to remove channel effect
		features = (features - np.mean(features, axis=0))/(np.std(features, axis=0)+1e-8)

	return features

def standardized_labels(params, is_train, is_disyllable):

	vars_dir = 'vars'
	filename = 'label_mean_std_di.npy' if is_disyllable else 'label_mean_std_mono.npy' 
	filepath = join(vars_dir, filename)
	os.makedirs(vars_dir, exist_ok=True)
	
	# find mean of each feature in each timestep
	if is_train:
		mean = np.mean(params, axis=0)
		std = np.std(params, axis=0)
		np.save(filepath,np.array([mean, std]))
	else:
		if os.path.isfile(filepath):
			mean_std = np.load(filepath).tolist()
			mean = np.array(mean_std[0])
			std = np.array(mean_std[1])
		else:
			raise ValueError('File %s doest exist'%vars_dir)
	# normalize each feature by its mean and plus small value to prevent value of zero
	return (params - mean)/(std+1e-6)

def destandardized_label(params, is_disyllable):
	vars_dir = 'vars'
	filename = 'label_mean_std_di.npy' if is_disyllable else 'label_mean_std_mono.npy' 
	filepath = join(vars_dir, filename)
	if os.path.isfile(filepath):
		mean_std = np.load(filepath).tolist()
		mean = mean_std[0]
		std = mean_std[1]
	return (params*std)+mean

def min_max_scale_transform(params, is_train, is_disyllable, feature_range=(-1, 1)):

	vars_dir = 'vars'
	filename = 'labe_scaler_di.joblib' if is_disyllable else 'labe_scaler_mono.joblib' 
	filepath = join(vars_dir, filename)
	os.makedirs(vars_dir, exist_ok=True)

	if is_train:
		scaler = MinMaxScaler(feature_range=feature_range).fit(params)
		dump(scaler, filepath)
	else:
		if os.path.isfile(filepath):
			scaler = load(filepath) 
		else:
			raise ValueError('File %s doest exist'%vars_dir)

	params = scaler.transform(params)

	return params

def min_max_descale_labels(params, is_disyllable):
	vars_dir = 'vars'
	filename = 'labe_scaler_di.joblib' if is_disyllable else 'labe_scaler_mono.joblib' 
	filepath = join(vars_dir, filename)
	print(filepath)

	if os.path.isfile(filepath):
		scaler = load(filepath) 
	else:
		raise ValueError('File %s doest exist'%vars_dir)

	return scaler.inverse_transform(params)

def descale_labels(scale_params):
	# the function is called before the parameters was added.
	ph = np.delete(param_high, DEL_PARAMS_LIST , axis=0)
	pl = np.delete(param_low, DEL_PARAMS_LIST , axis=0)
	return scale_params*(ph - pl) + pl

def scale_label_back(scale_params):
	# the function is same as descale_labels but use in preprocess when label normalize is set to 4
	return scale_params*(param_high - param_low) + param_low

def add_params(params):
	'''
	This function remove WC, TRX, TRY, and MS1,2,3 paramter
	'''
	# Add JX
	params = np.insert(params, 2, 0.0, axis=1) 
	# WC param
	params = np.insert(params, 8, 0.0, axis=1) 
	# TRX param
	TRX = params[:,10]*0.9380 - 5.1100
	params = np.insert(params, 15, TRX, axis=1) 
	# TRY param
	TRY = params[:,9]*0.8310 -3.0300
	params = np.insert(params, 16, TRY, axis=1) 
	# MS param
	params = np.insert(params, 21, -0.05, axis=1) 
	params = np.insert(params, 22, -0.05, axis=1) 
	params = np.insert(params, 23, -0.05, axis=1) 
	return params

def transform_VO(labels):
	'''
	Use this function after addd param function
	'''
	VO = [-1.0, -0.10, 0.050]

	for item in labels:
		item[7:8] = VO[np.argmin([abs(VO[0]-item[7]), abs(VO[1]-item[7]), abs(VO[2]-item[7])])]
	return labels

def label_imputation(params_descale):
	# Add JX
	params_descale[:,2] = 0
	# adjust VO
	# params_descale[params[:,7] > -0.5,7] = 0.05 
	# params_descale[params[:,7] <= -0.5,7] = -1.0
	# WC param
	params_descale[:,8] = 0
	# TRX param
	params_descale[:,15] = params_descale[:,10]*0.9380 - 5.1100
	# TRY param
	params_descale[:,16] = params_descale[:,9]*0.8310 -3.0300
	# MS param
	params_descale[:,21] = -0.05
	params_descale[:,22] = -0.05
	params_descale[:,23] = -0.05
	return params_descale

def detransform_label(label_mode, y_pred, is_disyllable):
	if label_mode == 1:
		params = transform_VO(add_params(destandardized_label(y_pred, is_disyllable)))
	elif label_mode == 2:
		params = transform_VO(add_params(descale_labels(y_pred)))
	elif label_mode == 3:
		params = add_params(y_pred)
		params = min_max_descale_labels(params, is_disyllable)
		params = label_imputation(params)

	return params

def show_result(history, save_file): 

	epochs = range(1, int(len(history.history['loss'])) + 1)   

	# Define a subplot 
	fig, axs = plt.subplots(3,1,figsize=(8,16))

	# Plot loss
	loss_plot = axs[0]

	loss_plot.plot(epochs, history.history['loss'], 'c--', label='Training Loss')
	loss_plot.plot(epochs, history.history['val_loss'], 'b', label='Validation Loss')
	loss_plot.annotate('Validate\n%0.4f' % history.history['val_loss'][-1], xy=(1, history.history['val_loss'][-1]), xytext=(8, 0), 
				 xycoords=('axes fraction', 'data'), textcoords='offset points')
	loss_plot.set_title('Training and Validation Loss')
	loss_plot.set_xlabel('Epochs')
	loss_plot.set_ylabel('Loss')
	loss_plot.legend()
	loss_plot.xaxis.set_major_locator(MaxNLocator(integer=True))

	# Plot accuracy
	met_plot = axs[1]

	met_plot.plot(epochs, history.history['rmse'], 'c--', label='Training RMSE')
	met_plot.plot(epochs, history.history['val_rmse'], 'b', label='Validation RMSE')
	met_plot.annotate('Validate\n%0.4f' % history.history['val_rmse'][-1], xy=(1, history.history['val_rmse'][-1]), xytext=(8, 0), 
				 xycoords=('axes fraction', 'data'), textcoords='offset points')
	met_plot.set_title('Training and Validation RMSE')
	met_plot.set_xlabel('Epochs')
	met_plot.set_ylabel('RMSE')
	met_plot.legend()
	met_plot.xaxis.set_major_locator(MaxNLocator(integer=True))
	
	r2_plot = axs[2]
	r2_plot.plot(epochs, history.history['R2'], 'c--', label='Training R2')
	r2_plot.plot(epochs, history.history['val_R2'], 'b', label='Validation R2')
	
	r2_plot.annotate('Validate\n%0.4f' % history.history['val_R2'][-1], xy=(1, history.history['val_R2'][-1]), xytext=(8, 0), 
				 xycoords=('axes fraction', 'data'), textcoords='offset points')
	r2_plot.set_title('Training and Validation R2')
	r2_plot.set_xlabel('Epochs')
	r2_plot.set_ylabel('R2')
	r2_plot.legend()
	r2_plot.xaxis.set_major_locator(MaxNLocator(integer=True))

	fig.savefig(save_file)
	plt.close()

# Compute RMSE
def compute_rmse(actual,pred, axis=1):
	# Compute RMSE by row (axis=1) result in rmse of each data
	# Compute RMSE by column (axis=0) result in rmse of each label
	return np.sqrt((np.square(actual - pred)).mean(axis=axis))

def total_rmse(actual,pred):
	return np.sqrt((np.square(actual - pred)).mean(axis=0)).mean(axis=0)

# Visualize RMSE
def rmse_distribution(rmse, save_file, title):

	fig, axs = plt.subplots(1,2, figsize=(15,4))
	sns.boxenplot(rmse, ax=axs[0])
	sns.boxplot(rmse,ax=axs[1])
	axs[0].set_title('%s'%title)
	axs[1].set_title('Boxplot %s'%title)
	fig.savefig(save_file)
	plt.close()

# compute pearson correlation
def compute_pearson_correlation(actual,pred):

	t_actual = np.transpose(actual)
	t_pred = np.transpose(pred)

	corr = []
	for i in range(len(t_actual)):
		corr.append(pearsonr(t_actual[i],t_pred[i])[0])
	return np.array(corr)

# plot scatter plot
def plot_scatter(actual,pred, save_dir):

	os.makedirs(save_dir ,exist_ok=True)

	t_actual = np.transpose(actual)
	t_pred = np.transpose(pred)

	for i in range(len(t_actual)):
		plt.scatter(t_actual[i],t_pred[i])
		plt.ylabel('Target')
		plt.xlabel('Predict')
		plt.title('Scatter plot %s'%param_name[i])
		plt.savefig(save_dir+'/corr_%s.png'%param_name[i])
		plt.close()

# Compute R2 using Sci-kit learn
def compute_R2(actual,pred, multioutput='raw_values'):
	return r2_score(actual,pred, multioutput=multioutput)

def compute_AdjustR2(actual,pred, multioutput='raw_values'):
	R2 = r2_score(actual,pred, multioutput=multioutput)
	return 1-(1-R2)*(int(pred.shape[0])-1)/(int(pred.shape[0])-int(pred.shape[1])-1)


# Calculate formant
def get_formant(sound_sets, save_dir, praatEXE, label, is_disyllable = False):
	
	F1 = []
	F2 = []
	F3 = []

	formantsPath = os.path.abspath(os.path.join(".", save_dir,'analyse_format_%s'%label))
	os.makedirs(formantsPath, exist_ok=True)

	for idx, file in enumerate(sound_sets):

		formantData = np.array(praat_scripts.getFormants(praatEXE,
														os.path.abspath(os.path.join(".", file)),
														os.path.join(formantsPath, "%s_format_%s.txt"%(label,idx)),
														5500))

		if is_disyllable:

			transition_point = int(formantData.shape[0]/2)

			F1.append(formantData[:transition_point,1])
			F1.append(formantData[transition_point:,1])
			F2.append(formantData[:transition_point,2])
			F2.append(formantData[transition_point:,2])
			F3.append(formantData[:transition_point,3])
			F3.append(formantData[transition_point:,3])
		else:
			F1.append(formantData[:,1])
			F2.append(formantData[:,2])
			F3.append(formantData[:,3])

	F1 = np.array(F1)
	F2 = np.array(F2)
	F3 = np.array(F3)
		
	np.save(arr=F1, file=os.path.join(save_dir,'%s_F1.npy'%label))
	np.save(arr=F2, file=os.path.join(save_dir,'%s_F2.npy'%label))
	np.save(arr=F3, file=os.path.join(save_dir,'%s_F3.npy'%label))
		
	return F1, F2, F3

def padding_and_trimming_each(actual,pred):
	audio_length = pred.shape[0]
	return np.array([data[:pred[idx].shape[0]] if data.shape[0] >= pred[idx].shape[0] else np.pad(data, (0, max(0, pred[idx].shape[0] - data.shape[0])), "constant", constant_values =1) for idx, data in enumerate(actual)])

def padding_col(data):
	max_len = len(max(data, key=len))
	return np.array([item[:max_len] if item.shape[0] >= max_len else np.pad(item, (0, max(0, max_len - item.shape[0])), "constant", constant_values =1) for idx, item in enumerate(data)])

def calculate_relative_error(actual,pred):
	a = np.absolute(np.subtract(actual,pred))
	b = np.absolute(actual)
	c = np.divide(a,b)
	d = np.mean(c, axis=1)
	return d*100

def compute_formant_relative_error(target_sound, estimate_sound, formant_dir, praat_exe, is_disyllable):
	

	est_F1, est_F2, est_F3 = get_formant(estimate_sound, formant_dir, praat_exe, label='estimated', is_disyllable=is_disyllable)
	act_F1, act_F2, act_F3 = get_formant(target_sound, formant_dir, praat_exe, label='actual', is_disyllable=is_disyllable)
	
	act_F1 = padding_and_trimming_each(act_F1, est_F1)
	act_F2 = padding_and_trimming_each(act_F2, est_F2)
	act_F3 = padding_and_trimming_each(act_F3, est_F3)

	act_F1 = padding_col(act_F1)
	act_F2 = padding_col(act_F2)
	act_F3 = padding_col(act_F3)

	est_F1 = padding_col(est_F1)
	est_F2 = padding_col(est_F2)
	est_F3 = padding_col(est_F3)

	F1_re = calculate_relative_error(act_F1, est_F1)
	F2_re = calculate_relative_error(act_F2, est_F2)
	F3_re = calculate_relative_error(act_F3, est_F3)

	return F1_re, F2_re, F3_re

def export_format_csv(label_name, formant_dir, disyllable = False):

	act_F1 = np.load(os.path.join(formant_dir,'actual_F1.npy'))
	act_F2 = np.load(os.path.join(formant_dir,'actual_F2.npy'))
	act_F3 = np.load(os.path.join(formant_dir,'actual_F3.npy'))
	est_F1 = np.load(os.path.join(formant_dir,'estimated_F1.npy'))
	est_F2 = np.load(os.path.join(formant_dir,'estimated_F2.npy'))
	est_F3 = np.load(os.path.join(formant_dir,'estimated_F3.npy'))

	act_F1 = padding_col(act_F1).mean(axis=1)
	act_F2 = padding_col(act_F2).mean(axis=1)
	act_F3 = padding_col(act_F3).mean(axis=1)

	est_F1 = padding_col(est_F1).mean(axis=1)
	est_F2 = padding_col(est_F2).mean(axis=1)
	est_F3 = padding_col(est_F3).mean(axis=1)

	df = pd.DataFrame(data={'Label':np.concatenate((label_name,label_name), axis=0), 
		'F1':np.concatenate((act_F1,est_F1), axis=0), 
		'F2':np.concatenate((act_F2,est_F2), axis=0), 
		'F3':np.concatenate((act_F3,est_F3), axis=0), 
		'Target':len(act_F1)*[1] +  len(est_F1)*[0]})
	if disyllable:
		df['Label'], df['syllable'] = df['Label'].str.split('-', 1).str
	df.to_csv(os.path.join(formant_dir,'data_point.csv'), index=False)

def plot_spectrogram(target_sig, predict_sig, save_file, color='binary'):
	plt.xticks(fontsize=18)
	fig, ax = plt.subplots(2,1, figsize = (7,6))

	ax[0].specgram(target_sig, NFFT=256, Fs=2, Fc=0, noverlap=128,
				 cmap=color, xextent=None, pad_to=None, sides='default',
				 scale_by_freq=None, mode='default', scale='default')
	ax[0].set_title('Target sound')
	ax[0].set_xlabel('Time [ms]')
	ax[0].set_ylabel('Frequency [Hz]')
	ax[1].specgram(predict_sig, NFFT=256, Fs=2, Fc=0, noverlap=128,
				 cmap=color, xextent=None, pad_to=None, sides='default',
				 scale_by_freq=None, mode='default', scale='default')
	ax[1].set_title('Inversion sound')
	ax[1].set_xlabel('Time [ms]')
	ax[1].set_ylabel('Frequency [Hz]')
	fig.tight_layout()
	fig.savefig(save_file)
	plt.close()

def plot_wave(target_sig, predict_sig, samp_rate, save_file):
	t=np.linspace(0, len(target_sig)/samp_rate, num=len(target_sig))
	fig, ax = plt.subplots(2,1, figsize = (7,4))
	ax[0].plot(t,target_sig)
	ax[0].set_title('Target wave audio')
	ax[0].set_xlabel('Time [s]')
	ax[0].set_ylabel('Frequency [Hz]')
	ax[1].plot(t,predict_sig)
	ax[1].set_title('Inversion wave audio')
	ax[1].set_xlabel('Time [s]')
	ax[1].set_ylabel('Frequency [Hz]')
	fig.tight_layout()
	fig.savefig(save_file)
	plt.close()

def generate_visualize_spectrogram(target_sound, predict_sound, save_dir, color='Greys'):
	os.makedirs(save_dir, exist_ok=True)

	try:
		if target_sound.shape[0] != predict_sound.shape[0]:
			raise ValueError('Unequal size of target and inversion')

		for idx, target in enumerate(target_sound):
			target_sig, rate = librosa.load(target, mono=True, sr=16000)
			predict_sig, rate = librosa.load(predict_sound[idx], mono=True, sr=16000)
			save_file = os.path.join(save_dir,'spec_%s'%idx)+'.png'
			plot_spectrogram(target_sig, predict_sig, save_file, color=color)

	except Exception as e:
		print('Visualization (spectrogram) cannot be created')
		print(e)

def generate_visualize_wav(target_sound, predict_sound, save_dir):
	os.makedirs(save_dir, exist_ok=True)
	try:
		if target_sound.shape[0] != predict_sound.shape[0]:
			raise ValueError('Unequal size of target and inversion')

		for idx, target in enumerate(target_sound):
			target_sig, rate = librosa.load(target, mono=True, sr=16000)
			predict_sig, rate = librosa.load(predict_sound[idx], mono=True, sr=16000)
			save_file = os.path.join(save_dir,'spec_%s'%idx) + '.png'
			plot_wave(target_sig, predict_sig[:target_sig.shape[0]], rate, save_file)

	except Exception as e:
		print('Visualization (wave) cannot be created')
		print(e)

def get_log():

	experiment_log = os.path.join('result','log.txt')
	if os.path.isfile(experiment_log):
		log = open(experiment_log, 'a') 
	else:
		log = open(experiment_log, 'w')

	return log

def get_experiment_number():

	# Load checkpoint
	with open(join('vars','exp_num.txt'), "r") as num:
		return list(map(int, num.readlines()))[0]

def get_label_prep_mode(data_path):

	with open(join(data_path,'label_mode.txt'), "r") as num:
		return list(map(int, num.readlines()))[0]