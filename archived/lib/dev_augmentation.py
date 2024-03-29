import numpy as np
import librosa, os, math, random, itertools

def random_sampling_data(audio_data, labels, phonetic, random_n_sample):
	
	idx = np.random.choice(range(len(phonetic)) , size=random_n_sample, replace = False)
	audio_data_sub = np.array([audio_data[i] for i in idx])
	labels_sub = np.array([labels[i] for i in idx])
	phonetic_sub = np.array([phonetic[i] for i in idx])
	
	return audio_data_sub, labels_sub, phonetic_sub

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

def init_change_pitch(sr):
	def change_pitch(audio_data):
		return np.array([librosa.effects.pitch_shift(data, sr=sr, n_steps=np.random.uniform(low=-1.0,high=4)) for data in audio_data])
	return change_pitch