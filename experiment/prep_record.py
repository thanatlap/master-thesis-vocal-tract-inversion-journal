'''
Preprocess a group of recorded data file

'''
import numpy as np
import pandas as pd
import librosa, os, math
from math import ceil, floor
from librosa import feature
from os.path import join
from os import makedirs
import scipy

data_file_m = [
	['d_record_nat',16000],
	['d_record_aj',32250],
	['d_record_sorn',22050],
	['d_record_kring',16000],
	['d_record_in',22050],
	['d_record_blank',22050]]

data_file_f = [
	['d_record_pat',16000],
	['d_record_mom',22050],
	['d_record_sorn_sis',44100],
	['d_record_gam',22050],
	['d_record_plooktan',22050],
	['d_record_su',25000]]

data_nat_set = [
	['d_record_set_1',16000],
	['d_record_set_2',16000],
	['d_record_set_3',16000],
	['d_record_set_4',16000],
	['d_record_set_5',16000],
	['d_record_set_6',16000],
	['d_record_set_8',16000]]

def composed_data(data_list, output_folder):
	
	audio_data = []
	split_data = []
	output_path = '../data/d_records/{}'.format(output_folder)
	
	makedirs(output_path, exist_ok=True)
	
	for data_folder in data_list:
	
		data_path = '../data/d_records/{}'.format(data_folder[0])
		filepath = join(data_path,'sound_set.txt')
		with open(filepath, 'r') as f:
			data = np.array(f.read().split(','))
		record_data = np.array([librosa.load(join(data_path, file+'.wav'), sr=44100)[0] for file in data])
		audio_data.extend(record_data)
		split_point = pd.read_csv(join(data_path,'split_mark.csv'))['split'].values.tolist()
		split_data.extend(split_point)

	_ = [scipy.io.wavfile.write(join(output_path,'sound{}.wav'.format(idx)), 44100, sig) for idx, sig in enumerate(audio_data)]
	
	with open(join(output_path,'sound_set.txt'), 'w') as f:
		for i in range(len(audio_data)):
			if i == (len(audio_data)-1):
				f.write('sound{}'.format(i))
			else:
				f.write('sound{},'.format(i))

	audio_idx = np.arange(len(audio_data))
	pd.DataFrame({'sound':audio_idx, 'split': split_data}, index=None).to_csv(join(output_path,'split_mark.csv'), index=False)

def import_data(data_path, sample_rate):

	with open(join(data_path,'sound_set.txt'), 'r') as f:
		data = np.array(f.read().split(','))
	audio_paths = [join(data_path, file+'.wav') for file in data]
	audio_data = np.array([ librosa.load(file, sr=sample_rate)[0] for file in audio_paths ])

	return audio_data

def split_disyllable(audio_data, data_path, sample_rate):

	split_point = pd.read_csv(join(data_path,'split_mark.csv'))['split'].values
	split_audio_data = np.array([data[:ceil(split_point[i]*sample_rate)] if j == 0 else data[floor(split_point[i]*sample_rate):floor(0.9*len(data))] for i, data in enumerate(audio_data) for j in range(2)])
	return split_audio_data

def transform_audio_to_mfcc(audio_data, sample_rate):

	outputs = []
	for data in audio_data:
		mfcc = feature.mfcc(data, sr=sample_rate, n_mfcc = 13)
		outputs.append(np.swapaxes(np.concatenate((mfcc,feature.delta(mfcc),feature.delta(mfcc, order=2)),axis=0),0,1))
	return outputs

def cepstral_normalization(mfccs):

	ceps_mean = np.mean([np.mean(item, axis=0) for item in mfccs], axis=0)
	ceps_std = np.mean([np.std(item, axis=0) for item in mfccs], axis=0)
	return np.array([(item-ceps_mean)/(ceps_std+1e-8) for item in mfccs])

def zero_pad_mfcc(feature):

	length_file = join('vars','max_mfcc_length_di.txt')
	if not os.path.exists(length_file):
		raise ValueError('segment max_length_file_path not found')
	else:
		with open(length_file, "r") as max_length:
			max_length = list(map(int, max_length.readlines()))[0]

	def pad_data(vector, pad_width, iaxis, kwargs):
		vector[:pad_width[0]] = kwargs.get('padder', 0)

	return np.array([item[:max_length] if item.shape[0] > max_length else np.pad(item, [(max(0, max_length-item.shape[0]),0),(0,0)], pad_data) for item in feature])

def export_syllable(data_list, output_path):

	file = 'syllable_name.txt'
	filenames = ['../data/d_records/{}/{}'.format(folder[0], file) for folder in data_list]
	with open(join(output_path,file), 'w') as outfile:
		for idx, fname in enumerate(filenames):
			with open(fname) as infile:
				outfile.write(infile.read())
				if idx != (len(filenames)-1): outfile.write(',')

def prep_feature(data_path, sample_rate):

	audio_data = import_data(data_path, sample_rate)
	split_audio_data = split_disyllable(audio_data, data_path, sample_rate)
	mfccs = transform_audio_to_mfcc(split_audio_data, sample_rate)
	mfccs = cepstral_normalization(mfccs)
	mfccs = zero_pad_mfcc(mfccs)

	return mfccs

def prep_record(data_list, output_folder):
	
	output_path = '../data/d_records/{}'.format(output_folder)
	print('[INFO] Grouping data from list')
	composed_data(data_list, output_folder)
	print('[INFO] Preprocess data features')
	features = [mfcc for data_folder in data_list for mfcc in prep_feature('../data/d_records/{}'.format(data_folder[0]), data_folder[1])]
	print('[INFO] Export data features')
	makedirs(join(output_path,'prep_data'), exist_ok=True)
	np.save(arr=np.array(features), file=join(output_path,'prep_data','features.npy'))
	print('[INFO] Export syllable')
	export_syllable(data_list, output_path)

def main():

	# prep_record(data_file_m, 'record_m')
	# prep_record(data_file_f, 'record_f')
	# prep_record(data_file_m + data_file_f, 'record_all')
	# prep_record(data_file_m + data_file_f, 'record_all_ns')

	prep_record(data_file_m + data_file_f + data_nat_set , 'record_total_data')

if __name__ == '__main__':
	main()