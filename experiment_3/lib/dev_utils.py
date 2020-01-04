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

param_high = np.array([1, -3.5, 0, 0, 1, 4, 1, 1, 1, 4, 1, 5.5, 2.5, 4, 5, 2, 0, 1.4, 1.4, 1.4, 1.4, 0.3, 0.3, 0.3])
param_low = np.array([0,-6.0, -0.5, -7.0, -1.0, -2.0, 0, -0.1, 0, -3, -3, 1.5, -3.0, -3, -3, -4, -6, -1.4, -1.4, -1.4, -1.4, -0.05, -0.05, -0.05]) 
param_name = np.array(["HX","HY","JX","JA","LP","LD","VS","VO","WC","TCX","TCY","TTX","TTY","TBX","TBY","TRX","TRY","TS1","TS2","TS3","TS4","MA1","MA2","MA3"])

def cnn_reshape(features):
	return features.reshape((features.shape[0],features.shape[1],features.shape[2],1))

#Function for preprocessing data
def scale_labels(params):
	return (params - param_low)/(param_high - param_low)

def descale_labels(scale_params):
	return scale_params*(param_high - param_low) + param_low

def delete_WC_param(params):
	return np.delete(params, 8, axis=1)

def delete_params(params):
	'''
	This function remove WC, TRX, TRY, and MS1,2,3 paramter
	'''
	return np.delete(params, [8,15,16,21,22,23] , axis=1)

def add_WC_param(params):
	return np.insert(params, 8, 0.0, axis=1) 

def add_params(params):
	'''
	This function remove WC, TRX, TRY, and MS1,2,3 paramter
	'''
	# WC param
	params = add_WC_param(params)
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

def label_transform(labels, invert=False):
	return descale_labels(add_params(labels)) if invert else delete_params(scale_labels(labels))

# define a function to plot the result from training step
def show_result(history, save_file, history_tag = ['loss','val_loss','rmse','val_rmse'], metric_label='RMSE'): 
	
	# Print the result from the last epoch
	print('Last train evaluation: %.3f'%history.history[history_tag[2]][-1])
	print('Last validation evaluation: %.3f'%history.history[history_tag[3]][-1])
	
	loss = history.history[history_tag[0]]
	val_loss = history.history[history_tag[1]]
	
	metric = history.history[history_tag[2]]
	val_metric = history.history[history_tag[3]]
	
	epochs = range(1, len(loss) + 1)   
	
	# Define a subplot 
	fig, axs = plt.subplots(1,2,figsize=(15,4))
	
	# Plot loss
	loss_plot = axs[0]
	
	loss_plot.plot(epochs, loss, 'c--', label='Training Loss')
	loss_plot.plot(epochs, val_loss, 'b', label='Validation Loss')
	loss_plot.set_title('Training and Validation Loss')
	loss_plot.set_xlabel('Epochs')
	loss_plot.set_ylabel('Loss')
	loss_plot.legend()
	
	# Plot accuracy
	met_plot = axs[1]
	
	met_plot.plot(epochs, metric, 'c--', label='Training Metric')
	met_plot.plot(epochs, val_metric, 'b', label='Validation Metric')
	met_plot.set_title('Training and Validation Evaluation')
	met_plot.set_xlabel('Epochs')
	met_plot.set_ylabel(metric_label)
	met_plot.legend()

	fig.savefig(save_file)
	plt.close()

# Define an evaluation function to print the evaluation result
def evaluation_report(model,features,labels):
	
	# Calculate result
	result = model.evaluate(features,labels,verbose=False)
	# Predict 
	y_pred = model.predict(features)
	# Show report
	print("Loss: %s Metric: %s" %(result[0],result[1]))
	
	return y_pred, result

# Compute RMSE
def compute_rmse(actual,pred, axis=1):
	# Compute RMSE by row (axis=1) result in rmse of each data
	# Compute RMSE by column (axis=0) result in rmse of each label
	return np.sqrt((np.square(actual - pred)).mean(axis=axis))

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
			F1.append(formantData[:int(formantData.shape[0]/2),1])
			F1.append(formantData[int(formantData.shape[0]/2):,1])
			F2.append(formantData[:int(formantData.shape[0]/2),2])
			F2.append(formantData[int(formantData.shape[0]/2):,2])
			F3.append(formantData[:int(formantData.shape[0]/2),3])
			F3.append(formantData[int(formantData.shape[0]/2):,3])
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

def calculate_relative_error(actual,pred):
	a = np.absolute(np.subtract(actual,pred))
	b = np.absolute(actual)
	c = np.divide(a,b)
	d = np.mean(c, axis=1)
	return d*100

def compute_formant_relative_error(target_sound, estimate_sound, formant_dir, praat_exe, is_disyllable):
	

	est_F1, est_F2, est_F3 = get_formant(estimate_sound, formant_dir, praat_exe, label='estimated', is_disyllable=is_disyllable)
	act_F1, act_F2, act_F3 = get_formant(target_sound, formant_dir, praat_exe, label='actual', is_disyllable=is_disyllable)
	
	F1_re = calculate_relative_error(act_F1, est_F1)
	F2_re = calculate_relative_error(act_F2, est_F2)
	F3_re = calculate_relative_error(act_F3, est_F3)

	return F1_re, F2_re, F3_re

def export_format_csv(label_name, formant_dir, sel_point = 13, disyllable = False):

	act_F1 = np.load(os.path.join(formant_dir,'actual_F1.npy')).mean(axis=1)
	act_F2 = np.load(os.path.join(formant_dir,'actual_F2.npy')).mean(axis=1)
	est_F1 = np.load(os.path.join(formant_dir,'estimated_F1.npy')).mean(axis=1)
	est_F2 = np.load(os.path.join(formant_dir,'estimated_F2.npy')).mean(axis=1)

	df = pd.DataFrame(data={'Label':np.concatenate((label_name,label_name), axis=0), 'F1':np.concatenate((act_F1,est_F1), axis=0), 'F2':np.concatenate((act_F2,est_F2), axis=0), 'Target':len(act_F1)*[1] +  len(est_F1)*[0]})
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