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
import librosa, os, math, random, itertools, argparse, pickle
from librosa import feature
import matplotlib.pyplot as plt
from os.path import join
from sklearn.preprocessing import MinMaxScaler
from time import time
from datetime import datetime
from functools import partial
from joblib import dump, load

import lib.dev_utils as utils
import lib.dev_augmentation as dev_aug

TRAIN_MODE = 'training'
EVAL_MODE = 'eval'
PREDICT_MODE = 'predict'

np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)
np.random.seed(seed=42)

def import_data(data_path, mode, sample_rate):
	# import data, note that if mode=predict, labels is [].
	if mode == TRAIN_MODE:
		data = np.load(join(data_path,'dataset.npz'))
		audio_data = data['ns_audio_data']
		labels = data['ns_aggregate_param']
		phonetic = data['ns_aggregate_phonetic']
	else:
		phonetic = []
		if mode == PREDICT_MODE:
			with open(join(data_path,'sound_set.txt'), 'r') as f:
				data = np.array(f.read().split(','))
			audio_paths = [join(data_path, file+'.wav') for file in data]
			labels = []
		elif mode == EVAL_MODE:
			data = np.load(join(data_path, 'csv_dataset.npz'))
			audio_paths = [join(data_path, 'sound',file) for file in data['sound_sets'][0]]
			labels = data['syllable_params']
		# load audio
		audio_data = np.array([ librosa.load(file, sr=sample_rate)[0] for file in audio_paths ])
	return audio_data, labels, phonetic

# function to scale param
def speaker_minmax_scale(params, data_path, sid, mode):
	# if training, get simulated speaker
	if mode == TRAIN_MODE:
		s_param = np.load(join(data_path, 'simulated_speakers', 'speaker_param_s%s.npz'%sid))
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
	if mode == TRAIN_MODE:
		speaker_sid = np.load(join(data_path, 'dataset.npz'))['ns_sid']
	else:
		# else, the scale is performed using predefined speaker
		speaker_sid = [0 for i in labels]
	return np.array([ speaker_minmax_scale(label, data_path, speaker_sid[n], mode) for n, label in enumerate(labels) ])

def split_audio(audio_data, labels, phonetic, mode):
	'''
	For disyllabic data, the audio is being splited by half
	the label is being split
	'''
	# for others circumstance
	split_point = 0.5
	cut_end = 1.0

	# for aj record
	# split_point = 0.325
	# cut_end = 0.75

	# for mom record
	# split_point = 0.45
	# cut_end = 0.9

	# for kring record
	# split_point = 0.40
	# cut_end = 0.7

	# for sorn record
	# split_point = 0.4
	# cut_end = 0.9

	# for general record
	# split_point = 0.35
	# cut_end = 0.8

	# split audio
	split_audio_data = np.array([data[:math.ceil(split_point*len(data))] if j == 0 else data[math.floor(split_point*len(data)):math.floor(cut_end*len(data))] for i, data in enumerate(audio_data) for j in range(2)])
	# split labels for training set and eval set, for predicting data, it return empty set []
	split_labels = np.array([x for item in labels for x in item]) if mode in [TRAIN_MODE, EVAL_MODE] else []
	split_phonetic = np.array([x for item in phonetic for x in item]) if mode in [TRAIN_MODE] else []

	return split_audio_data, split_labels, split_phonetic


def get_mfcc_max_length(feature, is_train, is_disyllable):
	'''
	get max length of list item in array of array
	'''
	max_length_file_path = join('vars','max_mfcc_length_di.txt') if is_disyllable else join('vars','max_mfcc_length_mono.txt')

	if is_train:
		os.makedirs('vars', exist_ok = True)
		max_length = max([item.shape[0] for item in feature])
		f = open(max_length_file_path, 'w')
		f.write(str(max_length))
		f.close()
	else:
		if not os.path.exists(max_length_file_path):
			raise ValueError('segment max_length_file_path not found')
		with open(max_length_file_path, "r") as max_length:
			max_length = list(map(int, max_length.readlines()))[0]
	return max_length


def zero_pad_mfcc(feature, is_train, is_disyllable):
	'''
	This function take mfcc with (datasize, timeframe, feature) 
	and padd zero 
	'''
	# save for inference
	max_length = get_mfcc_max_length(feature, is_train, is_disyllable)
	def pad_data(vector, pad_width, iaxis, kwargs):
		vector[:pad_width[0]] = kwargs.get('padder', 0)

	print(max_length)

	return np.array([item[:max_length] if item.shape[0] > max_length else np.pad(item, [(max(0, max_length-item.shape[0]),0),(0,0)], pad_data) for item in feature])


def transform_audio_to_mfcc(audio_data, max_n_mfcc):

	outputs = []
	for data in audio_data:
		mfcc = feature.mfcc(data, sr=16000, n_mfcc = max_n_mfcc)
		outputs.append(np.swapaxes(np.concatenate((mfcc,feature.delta(mfcc),feature.delta(mfcc, order=2)),axis=0),0,1))
	return outputs


def shuffle(mfcc, labels, phonetic, audio_feature):

	shuffle_idx = np.random.permutation(phonetic.shape[0])

	mfcc_shuffle = np.array([mfcc[i] for i in shuffle_idx])
	labels_shuffle = np.array([labels[i] for i in shuffle_idx])
	phonetic_shuffle = np.array([phonetic[i] for i in shuffle_idx])
	audio_shuffle = np.array([audio_feature[i] for i in shuffle_idx])

	return mfcc_shuffle, labels_shuffle, phonetic_shuffle, audio_shuffle

def augmentation(audio_data, labels, phonetic, augment_samples, func):
	'''
	the function to perform data augmentation
	First, the data is being sampling both labels and audio data
	Next, the augmentation is performed
	Then, the data is being concated to original data (both audio and labels)
	'''
	audio_data_sub, labels_sub, phonetic_sub = dev_aug.random_sampling_data(audio_data, labels, phonetic, random_n_sample=augment_samples)
	aug_data = func(audio_data_sub)
	audio_data = np.concatenate((audio_data, aug_data), axis=0)
	labels = np.concatenate((labels, labels_sub), axis=0)
	phonetic = np.concatenate((phonetic, phonetic_sub), axis=0)

	return audio_data, labels, phonetic

def preprocess_pipeline(features, labels, phonetic, mode, is_disyllable, is_train, mfcc_n, X_shape=None): 

	if is_disyllable:
		print('[INFO] Spliting audio data for disyllabic')
		features, labels, phonetic = split_audio(features, labels, phonetic, mode=mode)

	if mode != 'predict':
		print('[INFO] Min Max Scale using it own min max, not a predefined')
		labels = utils.min_max_scale_transform(labels, is_train, is_disyllable)

	print('[INFO] Feature mode set to 4, transform axis, mfcc, d, dd')
	mfcc_features = transform_audio_to_mfcc(features, max_n_mfcc = mfcc_n)
	print('[INFO] Normalization in each cepstral + self-normalized')
	mfcc_features = utils.normalize_mfcc_by_mean_cepstral(mfcc_features, is_train, is_disyllable, mode, X_shape=X_shape)
	print('[INFO] Padding MFCC length')
	mfcc_features = zero_pad_mfcc(mfcc_features, is_train, is_disyllable)

	if mode == TRAIN_MODE:
		#shuffle training subset after preprocess
		mfcc_features, labels, phonetic, features = shuffle(mfcc_features, labels, phonetic, features)
	else: 
		features = []
		phonetic = []

	return mfcc_features, labels, phonetic, features

def split_dataset(audio_data, labels, phonetic, val_test_split_ratio):
	
	total_size = len(phonetic)
	train_size = int((1-val_test_split_ratio)*total_size)
	data_idx = range(len(phonetic)) 
	
	idx = np.random.choice(data_idx, size=train_size, replace = False)
	train_audio = np.array([audio_data[i] for i in idx])
	train_labels = np.array([labels[i] for i in idx])
	train_phonetic = np.array([phonetic[i] for i in idx])
	
	test_data_idx = [i for i in data_idx if i not in idx]
	test_idx = np.random.choice(test_data_idx, size=int((total_size - train_size)/2), replace = False)
	test_audio = np.array([audio_data[i] for i in test_idx])
	test_labels = np.array([labels[i] for i in test_idx])
	test_phonetic = np.array([phonetic[i] for i in test_idx])
	
	val_data_idx = [i for i in data_idx if (i not in idx) and (i not in test_idx)]
	val_audio = np.array([audio_data[i] for i in val_data_idx])
	val_labels = np.array([labels[i] for i in val_data_idx])
	val_phonetic = np.array([phonetic[i] for i in val_data_idx])
	
	return train_audio, train_labels, train_phonetic, test_audio, test_labels, test_phonetic, val_audio, val_labels, val_phonetic

def main(args):

	rand_seed = np.random.randint(10, size=5)
	print('[DEBUG] Check random seed {}'.format(rand_seed))
	start_time = time()
	timestamp = datetime.now().strftime("%Y %B %d %H:%M")
	print('[INFO] {}'.format(timestamp))
	# check if data path is existed or not
	if not os.path.exists(args.data_path):
		raise ValueError('[ERROR] Data path %s is not exist'%args.data_path)
	# check if mode is corrected or not
	if args.mode not in [TRAIN_MODE, EVAL_MODE, PREDICT_MODE]:
		raise ValueError('[ERROR] Preprocess mode %s is not match [training, eval, predict]'%args.mode)
	# check syllable mode
	if args.syllable.lower() not in ['mono','di']:
		raise ValueError('[ERROR] Preprocess mode %s is not match [mono, di]'%args.syllable)
	else:
		disyllable = True if args.syllable.lower() == 'di' else False
	# convert string to boolean
	is_augment = utils.str2bool(args.is_augment)

	print('[INFO] CONFIG DETAIL')
	print('--Sample rate: %s'%str(args.sample_rate))
	print('--Test size: %s'%str(args.split_size))
	print('--Applied augment: %s'%str(is_augment))
	print('--Augment ratio sample: %s'%str(args.augment_samples))
	print('--Sampling data size: %s'%str(args.subsampling))
	print('--MFCC coefficient number: {}'.format(args.mfcc_coef))

	print('[INFO] Importing data')
	audio_data, labels, phonetic = import_data(args.data_path, mode=args.mode, sample_rate=args.sample_rate)

	print('[INFO]\n--- Audio Shape: %s'%str(audio_data.shape))
	if args.mode == TRAIN_MODE: print('--- Labels Shape: %s'%str(labels.shape))
	if args.mode == TRAIN_MODE: print('--- Phonetic Shape: %s'%str(phonetic.shape))


	if args.mode == TRAIN_MODE: 

		# subsample data
		if args.subsampling:
			print('[INFO] Sampling Data of size: {}'.format(args.subsampling))
			idx = np.random.choice(range(0,audio_data.shape[0]), size=args.subsampling, replace =False)
			audio_data = audio_data[idx]
			labels = labels[idx]
			phonetic = phonetic[idx]
			print('[DEBUG] Feature: {}, Label: {}'.format(str(audio_data.shape, labels.shape)))
	
	# print('[INFO] Reduce length for testing')
	# audio_data = audio_data[:100]
	# labels = labels[:100]
	# phonetic = phonetic[:100]

	if args.mode != 'predict': 

		print('[INFO] Scale Back to Predefine Speaker')
		labels = utils.scale_label_back(scale_speaker_syllable(labels, args.data_path, args.mode))
	
	p_preprocess_pipeline = partial(preprocess_pipeline,
		mode=args.mode, 
		is_disyllable=disyllable, 
		is_train=False,
		data_path = args.data_path,
		mfcc_n = args.mfcc_coef,
		X_shape=None)

	# split data into train, test, validate subset if mode = TRAIN_MODE, else, evaluate and test
	if args.mode == TRAIN_MODE:

		print('[INFO] Split audio data into different subset')
		X_train, y_train, z_train, X_test, y_test, z_test, X_val, y_val, z_val = split_dataset(audio_data, labels, phonetic, val_test_split_ratio=args.split_size)
		# perform augmentation for training dataset (only X_train)
		num_X_train = X_train.shape[0]
		if is_augment:
			print('[INFO] Augmenting audio data')
			
			# compute number of sample being augment based on training subset
			augment_samples = int(args.augment_samples*num_X_train)
			p_augmentation = partial(augmentation, augment_samples=augment_samples)

			X_train, y_train, z_train = p_augmentation(audio_data=X_train, labels=y_train, phonetic = z_train, func=dev_aug.init_change_pitch(args.sample_rate))
			X_train, y_train, z_train = p_augmentation(audio_data=X_train, labels=y_train, phonetic = z_train, func=dev_aug.amplify_value)
			X_train, y_train, z_train = p_augmentation(audio_data=X_train, labels=y_train, phonetic = z_train, func=dev_aug.add_white_noise)

		X_train, y_train, z_train, _ = p_preprocess_pipeline(features=X_train, labels=y_train, phonetic = z_train,is_train=True, X_shape=num_X_train)
		X_test, y_test, z_test, audio_test = p_preprocess_pipeline(features=X_test, labels=y_test, phonetic = z_test)
		X_val, y_val, z_val, audio_val = p_preprocess_pipeline(features=X_val, labels=y_val, phonetic = z_val)

	else:
		# for evaluated and predict dataset
		features, labels, _, _ = p_preprocess_pipeline(features=audio_data, labels=labels, phonetic=phonetic)
		
	output_path = join(args.data_path, args.output_path) 
	# create output file
	os.makedirs(output_path, exist_ok=True)

	# export data
	print('[INFO] Exporting features and labels')
	if args.mode == TRAIN_MODE:
		np.savez(join(output_path, 'training_subsets.npz'), 
			X_train = X_train,
			y_train= y_train, 
			z_train=z_train,
			X_val = X_val,
			y_val = y_val,
			z_val = z_val,
			X_test = X_test,
			y_test = y_test,
			z_test = z_test,
			audio_test = audio_test,
			audio_val = audio_val)
	elif args.mode == EVAL_MODE:
		np.savez(join(output_path,'eval_dataset.npz'),
			labels= labels,
			features = features)
	else:
		np.save(arr=features, file=join(output_path,'features.npy'))

	total_time = time()-start_time
	print('[Time: %.3fs]'%total_time)

	log = open(join(output_path,'description.txt'),"w")
	log.write('Date %s\n'%timestamp)
	log.write('Mode: %s\n'%str(args.mode))
	log.write('Data_path: %s\n'%str(args.data_path))
	log.write('Syllable: %s\n'%str(args.syllable))
	log.write('Used Augmentation: %s\n'%str(is_augment))
	log.write('Output_path: %s\n'%str(args.output_path))
	log.write('Augment_Frac: %s\n'%str(args.augment_samples))
	log.write('Test size (in percent): %s\n'%str(args.split_size))
	log.write('MFCC coefficient number: {}\n'.format(args.mfcc_coef))
	log.write('Total time used: %s\n'%total_time)
	log.write('[DEBUG] Random Code: {}\n'.format(rand_seed))
	log.close()

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Data preprocessing")
	parser.add_argument("mode", help="preprocessing mode ['training', 'eval', 'predict']", type=str)
	parser.add_argument("data_path", help="data parent directory", type=str)
	parser.add_argument("syllable", help="is data disyllable or monosyllable ['mono','di']", type=str)
	parser.add_argument("output_path", help="output directory", type=str, default=None)
	parser.add_argument('--is_augment', dest='is_augment', default='True', help='proceed data augmentation', type=str)
	parser.add_argument("--augment_samples", help="data augmentation fraction from 0 to 1", type=float, default=0.25)
	parser.add_argument("--sample_rate", help="audio sample rate", type=int, default=16000)
	parser.add_argument("--subsampling", help="sample data size", type=int, default=None)
	parser.add_argument("--split_size", help="size of test and validate from training dataset in percent", type=float, default=0.3)
	parser.add_argument("--mfcc_coef", help="number of mfcc_coef", type=int, default=13)
	args = parser.parse_args()
	main(args)

	