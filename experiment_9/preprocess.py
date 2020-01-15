'''
The script take the npy data (audio filename) 
and output the features (audio data) and labels (param sets)
- Audio
This file segment the audio data into an eqaul length 
Then, augment the audio using various method.
Next, if the audio is disyllable, the audio is being split
Last, the audio is transfrom to mfcc and add the delta and delta-delta features
- Label
Label is scale by the speaker it belong (for training set)
For evalset, the speaker is scaled by predefine speaker
The predict dataset does not have the label

'''
import numpy as np
import librosa 
from librosa import feature
import matplotlib.pyplot as plt
import os
import math
from os.path import join
import random
import itertools
import argparse
from sklearn.model_selection import train_test_split
from datetime import datetime
from sklearn.utils import shuffle

import lib.dev_utils as utils


def import_data(data_path, mode):
	'''
	import data, if it is a predict data, it read sound_set directly
	if it is training and eval, it read from npz file
	'''
	if mode == 'predict':
		data = []
		with open(join(data_path,'sound_set.txt'), 'r') as f:
		    data = np.array(f.read().split(','))
		# path to audio data
		audio_paths = [join(data_path, file+'.wav') for file in data]
		labels = []
	else:
		# key for indexing the data from npz
		data_filename = 'dataset.npz' if mode == 'training' else 'testset.npz'
		sound_key = 'ns_sound_sets' if mode == 'training' else 'sound_sets'
		label_key = 'ns_param_sets' if mode == 'training' else 'param_sets'
		# load data
		data = np.load(join(data_path, 'npy', data_filename))
		audio_filename = data[sound_key]
		labels = data[label_key]
		# path to audio data
		audio_paths = [join(data_path,'sound', file) for file in audio_filename]
	return audio_paths, labels

def load_audio(audio_paths, sample_rate):
	'''
	load audio data into npy
	'''
	audio_data = np.array([ librosa.load(file, sr=sample_rate)[0] for file in audio_paths ])
	return audio_data


def random_sampling_data(audio_data, labels, random_n_sample):
	'''
	random sampling audio data for data augmentation 
	'''
	audio_data_sub, labels_sub = zip(*random.sample(list(zip(audio_data, labels)), int(random_n_sample)))
	return np.array(audio_data_sub), np.array(labels_sub)

def strech_audio(audio_data):
	'''
	strech (faster or slower) audio data which vary by 0.8 and 1.5
	'''
	return [ librosa.effects.time_stretch(data, rate=np.random.uniform(low=0.8,high=1.1)) for data in audio_data]

def amplify_value(audio_data):
	'''
	amplify value by 1.5 to 3 time the original data
	'''
	return np.array([ data*np.random.uniform(low=1.5,high=3) for data in audio_data])

def add_white_noise(audio_data):
	'''
	add white noise
	'''
	return np.array([ data + np.random.uniform(low=0.001,high=0.01)*np.random.randn(data.shape[0]) for data in audio_data ])

def random_crop_out(audio_data):
    '''
    random crop some part of the data out
    '''
    n_cropout = [ np.random.choice([1,2,3], p=[0.6, 0.2, 0.2]) for data in range(len(audio_data))]

    aug_data = []
    for idx, data in enumerate(audio_data):
        for i in range(n_cropout[idx]):
            start = int(np.random.uniform(0, data.shape[0]))
            crop_length = int(np.random.uniform(0.1, 0.25)*data.shape[0])
            end = crop_length + start
            if end > data.shape[0]:
                end = data.shape[0]
            data[start:end] = [0]*int(end-start)
        aug_data.append(data)
    return np.array(aug_data)

def augmentation(audio_data, labels, augment_samples, func):
	'''
	the function to perform data augmentation
	First, the data is being sampling both labels and audio data
	Next, the augmentation is performed
	Then, the data is being concated to original data (both audio and labels)
	'''
	audio_data_sub, labels_sub = random_sampling_data(audio_data, labels, random_n_sample=augment_samples)
	aug_data = func(audio_data_sub)
	audio_data = np.concatenate((audio_data, aug_data), axis=0)
	labels = np.concatenate((labels, labels_sub), axis=0)
	return audio_data, labels

def load_from_export(filename):
	# load from export
	return np.load(filename)

# function to scale param
def speaker_minmax_scale(params, data_path, sid, mode):
	# if training, get simulated speaker
	if mode == 'training':
		s_param = np.load(join(data_path, 'speaker_sim', 'speaker_param%s.npz'%sid))
	else:
		# else, get predefine speaker from lib/templates
		s_param = np.load(join('lib', 'templates', 'speaker_param.npz'))
	p_low = s_param['low']
	p_high = s_param['high']
	return (params - p_low)/(p_high - p_low)

def scale_speaker_syllable(labels, data_path, mode):
	'''
	scale vocaltract parameter by max and min parameter that that speaker sid
	'''
	# For training dataset, the scale is performed using simulated speaker  
	if mode == 'training':
		speaker_sid = np.load(join(data_path, 'npy', 'dataset.npz'))['ns_sid']
	else:
		# else, the scale is performed using predefined speaker
		speaker_sid = [0 for i in labels]
	return np.array([ speaker_minmax_scale(label, data_path, speaker_sid[n], mode) for n, label in enumerate(labels) ])

def split_audio(audio_data, labels, mode):
	'''
	For disyllabic data, the audio is being splited by half
	the label is being split
	'''
	# split audio
	split_audio = np.array([data[:math.ceil(0.5*len(data))] if j == 0 else data[math.floor(0.5*len(data)):] for i, data in enumerate(audio_data) for j in range(2)])
	# split labels for training set and eval set, for predicting data, it return empty set []
	repeat_labels = np.array([x for item in labels for x in item]) if mode in ['training','eval'] else []
	return split_audio, repeat_labels

def get_audio_max_length(audio_data, mode, is_train, is_disyllable):
	'''
	get the maximum audio length from a train dataset and save to audio_length.npy
	'''
	audio_length_file = join('vars','audio_length_di.txt') if is_disyllable else join('vars','audio_length_mono.txt')
	# for training set, the length is save
	if mode in ['training'] and is_train:
		os.makedirs('vars', exist_ok = True)

		audio_length = len(max(audio_data, key=len))
		f = open(audio_length_file, 'w')
		f.write(str(audio_length))
		f.close()
	else:
		if not os.path.exists(audio_length_file):
			raise ValueError('segment length not found')
		with open(audio_length_file, "r") as audio_length:
			audio_length = list(map(int, audio_length.readlines()))[0]
	return audio_length

def zero_padding_audio(audio_data, mode, is_disyllable, is_train):
	audio_length = get_audio_max_length(audio_data, mode, is_train, is_disyllable)
	audio_length = math.floor(audio_length*0.5) if is_disyllable else audio_length
	return np.array([data[:audio_length] if data.shape[0] > audio_length else np.pad(data, (0, max(0, audio_length - data.shape[0])), "constant") for data in audio_data])



def transfrom_mfcc(audio_data, sample_rate):
	'''
	Transform audio feature into mfcc
	'''
	def wav2mfcc(data):
		max_n_mfcc = 13
		mfcc = librosa.feature.mfcc(data, sr=sample_rate, n_mfcc = max_n_mfcc)[:max_n_mfcc] # experiment with first mfcc features
		return mfcc

	return np.array([wav2mfcc(data) for data in audio_data])

def transform_delta(mfcc):
	'''
	Compute delta and delta-delta of the mfcc feature
	'''
	return np.concatenate((mfcc,feature.delta(mfcc),feature.delta(mfcc, order=2)),axis=1)

def standardize_mfcc(features, is_train, is_disyllable):

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
			mean = mean_std[0]
			std = mean_std[1]
		else:
			raise ValueError('File %s doest exist'%vars_dir)
	# normalize each feature by its mean and plus small value to 
	# prevent value of zero
	# features -= (np.mean(features, axis=0) + 1e-8)
	features = (features - mean)/std
	
	return features 

def preprocess_pipeline(features, labels, mode, is_disyllable, sample_rate, is_train, label_prep_mode, data_path=None):
	
	if is_disyllable:
		# split audio data for disyllable, note that if mode=predict, labels is [].
		print('[INFO] Spliting audio data for disyllabic')
		features, labels = split_audio(features, labels, mode=mode)

	if mode != 'predict':
		print('[INFO] Remove label param having std < 0.05')
		labels = utils.delete_params(labels)
		if label_prep_mode == 1:
			print('[INFO] Standardized labels')
			labels = utils.standardized_labels(labels, is_train, is_disyllable)
		elif label_prep_mode == 2:
			labels = scale_speaker_syllable(labels, data_path, mode)

	print('[INFO] Padding audio length')
	features = zero_padding_audio(features, mode, is_disyllable, is_train)
	# get mfcc
	print('[INFO] Transforming audio data to MFCC')
	features = transfrom_mfcc(features, sample_rate=sample_rate)
	# get delta and delta-delta
	print('[INFO] Adding delta and delta-delta')
	features = transform_delta(features)
	# Normalize MFCCs 
	print('[INFO] Normalization in each timestep')
	features = standardize_mfcc(features, is_train, is_disyllable)
	# swap dimension to (data, timestamp, features)
	print('[INFO] Swap axis to (data, timestamp, features)')
	features = np.swapaxes(features ,1,2)

	if mode == 'training':
		#shuffle training subset after preprocess
		features, labels = shuffle(features, labels)

	return features, labels

def main(args):
	# check if data path is existed or not
	if not os.path.exists(args.data_path):
		raise ValueError('[ERROR] Data path %s is not exist'%args.data_path)
	# check if mode is corrected or not
	if args.mode not in ['training', 'eval', 'predict']:
		raise ValueError('[ERROR] Preprocess mode %s is not match [training, eval, predict]'%args.mode)
	# check syllable mode
	if args.syllable.lower() not in ['mono','di']:
		raise ValueError('[ERROR] Preprocess mode %s is not match [mono, di]'%args.mode)
	# store value to disyllable 
	disyllable = True if args.syllable.lower() == 'di' else False

	# check label_normalize 
	if args.label_normalize not in [1,2]:
		raise ValueError('[ERROR] Preprocess mode %s is not match [1: standardized, 2: min-max]'%args.mode)

	print('[INFO] Test size: %s'%str(args.split_size))
	print('[INFO] Applied augment: %s'%str(args.is_augment))
	print('[INFO] Augment ratio sample: %s'%str(args.augment_samples))
	print('[INFO] Sample rate: %s'%str(args.sample_rate))
	print('[INFO] label normalize mode: %s'%str(args.label_normalize))

	# import data, note that if mode=predict, labels is [].
	print('[INFO] Importing data')
	audio_paths, labels = import_data(args.data_path, mode=args.mode)
	print('[INFO] Loading audio and labels data')
	audio_data = load_audio(audio_paths, args.sample_rate)
	print('[INFO] Audio Shape: %s'%str(audio_data.shape))
	
	# split data into train, test, validate subset if mode = 'training', else, evaluate and test
	if args.mode == 'training':

		# compute testing and validating size
		split_size = int(args.split_size*audio_data.shape[0])
		print('[DEBUG] split_size: %s'%str(split_size))

		print('[INFO] Split audio data into different subset')
		X_train, X_test, y_train, y_test = train_test_split(audio_data, labels, test_size = split_size, random_state=0)
		X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size =split_size, random_state=0)
		# perform augmentation for training dataset (only X_train)
		if args.is_augment:
			print('[INFO] Augmenting audio data')

			# compute number of sample being augment based on training subset
			augment_samples = int(args.augment_samples*X_train.shape[0])
			print('[DEBUG] augment_samples: %s'%str(augment_samples))

			X_train, y_train = augmentation(X_train, y_train, augment_samples=augment_samples, func=strech_audio)
			X_train, y_train = augmentation(X_train, y_train, augment_samples=augment_samples, func=amplify_value)
			X_train, y_train = augmentation(X_train, y_train, augment_samples=augment_samples, func=add_white_noise)
			X_train, y_train = augmentation(X_train, y_train, augment_samples=augment_samples, func=random_crop_out)

		X_train, y_train = preprocess_pipeline(X_train, y_train, 
			mode=args.mode, 
			is_disyllable=disyllable, 
			sample_rate=args.sample_rate,
			is_train=True,
			label_prep_mode = args.label_normalize,
			data_path = args.data_path)

		X_test, y_test = preprocess_pipeline(X_test, y_test, 
			mode=args.mode, 
			is_disyllable=disyllable, 
			sample_rate=args.sample_rate,
			is_train=False,
			label_prep_mode = args.label_normalize,
			data_path = args.data_path)

		X_val, y_val = preprocess_pipeline(X_val, y_val, 
			mode=args.mode, 
			is_disyllable=disyllable, 
			sample_rate=args.sample_rate,
			is_train=False,
			label_prep_mode = args.label_normalize,
			data_path = args.data_path)

	else:
		# for evaluated and predict dataset
		features, labels = preprocess_pipeline(audio_data, labels, 
			mode=args.mode, 
			is_disyllable=disyllable, 
			sample_rate=args.sample_rate,
			is_train=False,
			label_prep_mode = args.label_normalize,
			data_path = args.data_path)
		
	# export data
	print('[INFO] Exporting features and labels')

	# if output path is not specify
	if args.output_path == None:
		print('[INFO] default output directory is used')
		args.output_path = 'prep_data'

	#used output path with the same directory as data path
	# ex, d_dataset_1/prep_data/training_subsets.npz
	output_path = join(args.data_path, args.output_path) 
	# create output file
	os.makedirs(output_path, exist_ok=True)
	# export training dataset
	if args.mode == 'training':
		np.savez(join(output_path, 'training_subsets.npz'), 
			X_train = X_train,
			y_train= y_train, 
			X_val = X_val,
			y_val = y_val,
			X_test = X_test,
			y_test = y_test)
	elif args.mode == 'eval':
		np.savez(join(output_path,'eval_dataset.npz'),
			labels= labels,
			features = features)
	else:
		np.save(arr=features, file=join(output_path,'features.npy'))

	log.open(join(output_path, 'label_mode.txt'),"w")
	log.write(args.label_normalize)
	log.close()

	log = open(join(output_path,'description.txt'),"w")
	log.write('Date %s\n'%str(datetime.now().strftime("%Y-%B-%d %H:%M")))
	log.write('Mode: %s\n'%str(args.mode))
	log.write('Data_path: %s\n'%str(args.data_path))
	log.write('Syllable: %s\n'%str(args.syllable))
	log.write('Used Augmentation: %s\n'%str(args.is_augment))
	log.write('Output_path: %s\n'%str(args.output_path))
	log.write('Augment_Frac: %s\n'%str(args.augment_samples))
	log.write('Sample_Rate: %s\n'%str(args.sample_rate))
	log.write('Test size (in percent): %s\n'%str(args.split_size))
	log.write('Label normalize mode: %s\n'%str(args.label_normalize))
	log.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Data preprocessing")
	parser.add_argument("mode", help="preprocessing mode ['training', 'eval', 'predict']", type=str)
	parser.add_argument("data_path", help="data parent directory", type=str)
	parser.add_argument("syllable", help="is data disyllable or monosyllable ['mono','di']", type=str)
	parser.add_argument('--is_augment', dest='is_augment', default=True, help='proceed data augmentation', type=bool)
	parser.add_argument("--output_path", help="output directory", type=str, default=None)
	parser.add_argument("--augment_samples", help="data augmentation fraction from 0 to 1", type=float, default=0.6)
	parser.add_argument("--sample_rate", help="audio sample rate", type=int, default=16000)
	parser.add_argument("--label_normalize", help="label normalize mode [1: standardized, 2: min-max]", type=int, default=1)
	parser.add_argument("--split_size", help="size of test dataset in percent (applied to both val and test)", type=float, default=0.05)
	args = parser.parse_args()
	main(args)