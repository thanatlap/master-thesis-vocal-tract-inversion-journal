import numpy as np
import os
import pandas as pd
import csv
from os.path import join, exists
import ctypes
import multiprocessing as mp
import itertools
import math
from shutil import copyfile
import shutil 
import shutil

PARAM = 'PARAM'

def load_file_csv(file):
	if os.path.exists(file):
		df = pd.read_csv(file)
		return df
	else:
		raise ValueError('File %s not found!'%file)

def error_check(param_set):

	if PARAM not in param_set.columns:
		raise ValueError('[ERROR] %s column is not exist in a dataframe'%PARAM)

	if param_set.shape[1] != 25:
		raise ValueError('[ERROR] Incomplete column provided!, only %s given'%str(param_set.shape[1]))

def split_param_set(param_set):
	syllable_labels = param_set[PARAM].values
	syllable_params = param_set.drop([PARAM], axis=1).values
	param_names = param_set.drop([PARAM], axis=1).columns.values
	return syllable_labels, syllable_params, param_names

def import_data_from_csv(csv_path):
	data = load_file_csv(csv_path)
	error_check(data)
	return split_param_set(data)

def create_disyllabic_parameter(syllable_params, syllable_labels):
	# convert monosyllable to disyllable
	syllable_labels = ['%s%s'%(subset[0],subset[1]) for subset in itertools.permutations(syllable_labels, 2)]
	syllable_params = np.array([param for param in itertools.permutations(syllable_params, 2)])
	return syllable_params, syllable_labels

def convert_to_disyllabic(params):
	return params.reshape((int(params.shape[0]/2),2,24))

def export_syllable_name(syllable_labels, output_path):
	with open(join(output_path, 'syllable_name.txt'), 'w', newline='') as file:
		csv.writer(file).writerow(syllable_labels)

def create_speaker_file(syllable_params, param_names, output_path, speaker_header_file, speaker_tail_file, is_disyllable = False):

	# Create speaker folder
	speaker_path = join(output_path,'speaker')
	os.makedirs(speaker_path, exist_ok=True)

	# Read speaker template
	speaker_head = open(speaker_header_file,'r').read()
	speaker_tail = open(speaker_tail_file,'r').read()

	speaker_filenames = ["speaker%s.speaker"%n for n, _ in enumerate(syllable_params)]
	syllable_placeholder = ['aaa','bbb']
	# Loop through list
	for kdx, params in enumerate(syllable_params):

		file_path = join(speaker_path,speaker_filenames[kdx])
		with open(file_path, 'w') as f:
			f.write(speaker_head)
			if is_disyllable:
				# Loop through each parameter in the list
				for idx, pair_param in enumerate(params):
					f.write('<shape name="%s">\n'%syllable_placeholder[idx])
					for jdx, param in enumerate(pair_param):
						f.write('<param name="%s" value="%.2f"/>\n'%(param_names[jdx],param))
					f.write('</shape>\n')
			else:
				f.write('<shape name="%s">\n'%syllable_placeholder[0])
				for jdx, param in enumerate(params):
					f.write('<param name="%s" value="%.2f"/>\n'%(param_names[jdx],param))
				f.write('</shape>\n')
			f.write(speaker_tail)

	return speaker_filenames

def create_ges_file(syllable_params, output_path, is_disyllable):
	
	ges_path = join(output_path,'ges')
	os.makedirs(ges_path, exist_ok=True)

	ges_file = 'gesture_disyllable_template.ges' if is_disyllable else 'gesture_monosyllable_template.ges'
	ges_filenames = [ges_file]*len(syllable_params)
	shutil.copy('templates/'+ges_file, ges_path)

	return ges_filenames

def ges_to_wav(output_file_set, speaker_file_list, gesture_file, VTL_path, output):

	VTL = ctypes.cdll.LoadLibrary(VTL_path)

	for idx, output_file in enumerate(output_file_set):
		speaker_file_name = ctypes.c_char_p(str.encode(speaker_file_list[idx]))
		gesture_file_name = ctypes.c_char_p(str.encode(gesture_file[idx]))
		wav_file_name = ctypes.c_char_p(str.encode(output_file))
		feedback_file_name = ctypes.c_char_p(b'feedback.txt')

		failure = VTL.vtlGesToWav(speaker_file_name,  # input
								  gesture_file_name,  # input
								  wav_file_name,  # output
								  feedback_file_name)  # output
	output.put(failure)
	

def generate_sound(speaker_filenames, ges_filenames, output_path,
	VTL_path = 'VTL/VocalTractLabApi.dll',
	njob = 4):

	'''
	return the list of an output filename 
	'''
	# Create speaker folder
	os.makedirs(join(output_path,'sound'), exist_ok=True)

	sound_sets = [join('sound',"sound%s.wav"%str(x)) for x,_ in enumerate(speaker_filenames)]
	output_file_set = [join(output_path, sound) for sound in sound_sets]
	speaker_file_set = [join(output_path, 'speaker', file) for file in speaker_filenames] 
	ges_file_set = [join(output_path, 'ges', file) for file in ges_filenames]

	# Start multiprocess
	# Define an output queue
	output = mp.Queue()

	processes = []
	for i in range(njob):
		start = i*int(len(sound_sets)/njob)
		end = (i+1)*int(len(sound_sets)/njob) if i != njob-1 else len(sound_sets)
		# Setup a list of processes that we want to run
		processes.append(mp.Process(target=ges_to_wav, args=(output_file_set[start:end], speaker_file_set[start:end], ges_file_set[start:end] ,VTL_path, output)))

	# Run processes
	for p in processes:
		p.start()

	# Exit the completed processes
	for p in processes:
		p.join()

	failures = [output.get() for p in processes]
	if any(failures) != 0: raise ValueError('Error at file: ',failures)
	os.remove('feedback.txt')
	return sound_sets