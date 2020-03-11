import numpy as np
import pandas as pd
import os
from os.path import join
import ctypes
import multiprocessing as mp
from time import time
import librosa
from librosa import feature
import itertools
import math
from shutil import copyfile
import shutil 

def invert_to_predefine_scale(params):
	s_param = np.load(join('lib', 'templates', 'speaker_param.npz'))
	p_low = s_param['low']
	p_high = s_param['high']
	return [(param - p_low)/(p_high - p_low) for param in params]

def read_audio_path(data_path):
	data = []
	with open(join(data_path,'sound_set.txt'), 'r') as f:
		data = np.array(f.read().split(','))
	# path to audio data
	return np.array([join(data_path, file+'.wav') for file in data])

def extract_duration(data_path):
	'''
	extract duration for data generating after prediction
	return a list of list [[sound1.wav, 1.2], [sound2.wav, 0.92]]
	'''

	audio_paths = read_audio_path(data_path)
	return np.array([ [file,librosa.get_duration(librosa.load(file)[0])] for file in audio_paths ])


def convert_to_disyllabic_parameter(params):
	'''
	convert monosyllable to disyllable
	used when convert the result from the model
	'''
	return params.reshape((int(params.shape[0]/2),2,24))

def create_disyllabic_parameter(params, syllable_name=None):

	'''
	for disyllable to create a parameter set from raw parameter set
	this function return param of [[p1,p2]], each p is a set of vocaltract parameter (24 params)
	'''
	# convert monosyllable to disyllable
	if syllable_name is not None:
		test_label_name = ['%s;1'%item if i == 0 else '%s;2'%item for subset in itertools.permutations(syllable_name, 2) for i, item in enumerate(subset)]
	else:
		# return empty list [] if syllable_name is not specify 
		test_label_name = []
	di_params = np.array([param for param in itertools.permutations(params, 2)])
	return di_params, test_label_name

def create_speaker_file(param_sets,output_dir,
	speaker_header_file = 'lib/templates/speaker_template_head.txt',
	speaker_tail_file = 'lib/templates/speaker_template_tail.txt',
	is_disyllable = False):

	# Check error
	if not os.path.isfile(speaker_header_file): 
		raise ValueError('Header file %s not exist'%speaker_header_file) 

	if not os.path.isfile(speaker_tail_file): 
		raise ValueError('Tail file %s not exist'%speaker_tail_file) 

	# Create speaker folder
	speaker_dir = join(output_dir,'speaker')
	os.makedirs(speaker_dir, exist_ok=True)

	# Read speaker template
	speaker_head = open(speaker_header_file,'r').read()
	speaker_tail = open(speaker_tail_file,'r').read()

	speaker_filenames = ["speaker%s.speaker"%n for n, _ in enumerate(param_sets)]

	# load vocaltract param name
	param_name = np.load('lib/templates/speaker_param.npz')['name']

	# Loop through list
	for idx, params in enumerate(param_sets):

		file_path = join(speaker_dir,speaker_filenames[idx])
		f = open(file_path, 'w')
		f.write(speaker_head)
		if is_disyllable:

			# Loop through each parameter in the list
			for idx, pair_param in enumerate(params):
				if idx == 0: 
					f.write('<shape name="aaa">\n')
				else:
					f.write('<shape name="bbb">\n')
				for jdx, param in enumerate(pair_param):
					f.write('<param name="%s" value="%.2f"/>\n'%(param_name[jdx],param))
				f.write('</shape>\n')

		else:
			f.write('<shape name="aaa">\n')
			for jdx, param in enumerate(params):
				f.write('<param name="%s" value="%.2f"/>\n'%(param_name[jdx],param))
			f.write('</shape>\n')

		# Close with tail part of the speaker file
		f.write(speaker_tail)
		f.close()

	return speaker_filenames

def ges_template_gen(ges_file, duration, is_disyllable):

	if is_disyllable:
		duration = duration/2

	f = open(ges_file, 'w')
	f.write('<gestural_score>\n<gesture_sequence type="vowel-gestures" unit="">\n>')
	f.write('<gesture value="aaa" slope="0.000000" duration_s="%.6f" time_constant_s="0.015000" neutral="0" />\n'%duration)
	if is_disyllable:
		f.write('<gesture value="bbb" slope="0.000000" duration_s="%.6f" time_constant_s="0.015000" neutral="0" />\n'%duration)
	f.write('</gesture_sequence>\n<gesture_sequence type="lip-gestures" unit="">\n</gesture_sequence>\n<gesture_sequence type="tongue-tip-gestures" unit="">\n')
	f.write('</gesture_sequence>\n<gesture_sequence type="tongue-body-gestures" unit="">\n</gesture_sequence>\n<gesture_sequence type="velic-gestures" unit="">\n')
	f.write('</gesture_sequence>\n<gesture_sequence type="glottal-shape-gestures" unit="">\n')
	f.write('<gesture value="modal" slope="0.000000" duration_s="%.6f" time_constant_s="0.015000" neutral="0" />\n'%duration)
	if is_disyllable:
		f.write('<gesture value="modal" slope="0.000000" duration_s="%.6f" time_constant_s="0.015000" neutral="0" />\n'%duration)
	f.write('</gesture_sequence>\n')
	f.write('<gesture_sequence type="f0-gestures" unit="st">\n')
	f.write('<gesture value="83.00000" slope="0.000000" duration_s="0.01000" time_constant_s="0.030000" neutral="0"/>\n')
	f.write('<gesture value="84.00000" slope="0.000000" duration_s="%.5f" time_constant_s="0.030000" neutral="0"/>\n'%duration)
	if is_disyllable:
		f.write('<gesture value="84.00000" slope="0.000000" duration_s="%.5f" time_constant_s="0.030000" neutral="0"/>\n'%duration)
	f.write('</gesture_sequence>\n')
	f.write('<gesture_sequence type="lung-pressure-gestures" unit="dPa">\n')
	f.write('<gesture value="0.000000" slope="0.000000" duration_s="0.010000" time_constant_s="0.005000" neutral="0" />\n')
	f.write('<gesture value="9000.000000" slope="0.000000" duration_s="%.6f" time_constant_s="0.005000" neutral="0" />\n'%duration)
	if is_disyllable:
		f.write('<gesture value="9000.000000" slope="0.000000" duration_s="%.6f" time_constant_s="0.005000" neutral="0" />\n'%duration)
	f.write('</gesture_sequence> \n')
	f.write('</gestural_score>\n')
	f.close()

def create_ges_file(param_sets, output_dir, data_path = None, is_disyllable = False, mode='others'):
	
	ges_dir = join(output_dir,'ges')
	os.makedirs(ges_dir, exist_ok=True)

	if mode == 'predict':

		audio_duration = extract_duration(data_path)
		ges_filenames = ["ges%s.speaker"%n for n, _ in enumerate(param_sets)]
		
		for idx, param in enumerate(param_sets):
			file_path = join(ges_dir, ges_filenames[idx])
			ges_template_gen(file_path, float(audio_duration[idx][1]), is_disyllable)

	else:
		ges_file = 'gesture_disyllable_template.ges' if is_disyllable else 'gesture_monosyllable_template.ges'
		ges_filenames = [ges_file]*len(param_sets)
		shutil.copy('lib/templates/'+ges_file, ges_dir)

	return ges_filenames


def initiate_VTL(VTL_path):
	'''
	initialize VTL (VocalTractLab Application) object
	to generate sound from GES and SPEAKER file

	'''
	if os.path.exists(VTL_path):
		# Call VTL application
		VTL = ctypes.cdll.LoadLibrary(VTL_path)
	else:
		raise ValueError('Path %s does not exist'%VTL_path)
	
	return VTL

def ges_to_wav(output_file_set, speaker_file_list, gesture_file,VTL_path, output):

	VTL = initiate_VTL(VTL_path)

	start_time = time()
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
	

def generate_sound(speaker_filenames, ges_filenames, output_dir,
	VTL_path = 'generator/assets/VTL/VocalTractLabApi.dll',
	njob = 4):

	'''
	return the list of an output filename 
	'''
	# Create speaker folder
	sound_dir = join(output_dir,'sound')
	os.makedirs(sound_dir, exist_ok=True)

	sound_sets = ["sound%s.wav"%str(x) for x,_ in enumerate(speaker_filenames)]
	output_file_set = [join(sound_dir, sound) for sound in sound_sets]
	speaker_file_set = [join(output_dir, 'speaker', file) for file in speaker_filenames] 
	ges_file_set = [join(output_dir, 'ges', file) for file in ges_filenames]

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

def convert_param_to_wav(param_sets, output_dir, is_disyllable, data_path=None, mode='others'):
	'''
	Take param set and create an audio data. The data is store in the output folder
	The output folder consist of 1. sound, 2. speaker, 3. npy

	'''
	start = time()
	speaker_filenames = create_speaker_file(param_sets, output_dir, is_disyllable=is_disyllable)
	ges_filenames = create_ges_file(param_sets, output_dir, data_path=data_path, is_disyllable=is_disyllable, mode=mode)

	sound_sets = generate_sound(speaker_filenames, ges_filenames, output_dir)
	np.savez(join(output_dir ,'testset.npz'), 
		param_sets = param_sets,
		speaker_filenames = speaker_filenames,
		sound_sets = sound_sets)
	print('Successfully convert label to sound [Time: %.3fs]'%(time()-start))

