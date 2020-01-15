import ctypes
import os
from os.path import join
from glob import glob
import shutil
import numpy as np
import pandas as pd
import math
import itertools
import librosa
from scipy import spatial
from scipy.spatial.distance import euclidean
from time import time
from datetime import datetime
import multiprocessing as mp
import config as cf
import simulate_speaker as ss
import random_param as rparm 
from functools import partial

param_high = ss.adult_high
param_low = ss.adult_low
param_neutral = ss.adult_neutral


def initiate_VTL():
	'''
	initialize VTL (VocalTractLab Application) object
	to generate sound from GES and SPEAKER file

	'''
	if os.path.exists(cf.VTL_FILE):
		# Call VTL application
		VTL = ctypes.cdll.LoadLibrary(cf.VTL_FILE)
		version = ctypes.c_char_p(b'                                ')
		VTL.vtlGetVersion(version)
		# print('VTL version (compile date): "%s"' % version.value.decode())
	else:
		raise ValueError('Path %s does not exist'%cf.VTL_FILE)
	
	return VTL

def load_file_csv(file):
	'''
	Load data from csv file.
	'''
	if os.path.exists(file):
		return pd.read_csv(file).values
	else:
		raise ValueError('File %s not found!'%file)

def _rand_fn(from_idx, to_idx, predefine_params):
	'''
	randomize vocaltract parameter based on randomly from min and max param (0.01 prob) 
	and weight warping from predefined parameter (0.99 prob)
	'''
	if np.random.uniform(0, high=1) < 0.005:
		random_param = rparm.randomize_params(predefine_params, param_high, param_low, sampling_step=cf.SAMPLING_STEP)
	else:
		random_param = rparm.randomize_by_percent_change(predefine_params, from_idx, to_idx)
	return random_param

def random_fn(predefine_params):
	'''
	generate a vocal tract parameter set.
	'''
	rand_range = predefine_params.shape[0]
	idx_pair = 4 if cf.DI_SYLLABLE else 2

	# randomize param and check duplicate index
	while True:
		param_idx = [ np.random.randint(rand_range) for i in range(idx_pair)]
		if ((len(param_idx) - len(set(param_idx))) == 0):
			break

	while True:
		pair_params = [_rand_fn(param_idx[i*2], param_idx[i*2+1], predefine_params) for i in range(int(idx_pair/2))]
		# Check if the param is directly pick from default param since we will used this as a final testing set.
		if sum([0 if not (item in predefine_params.tolist()) else 1 for item in pair_params]) == 0:
			break

	# If generate disyllable, return a pair of param else, return only first pair
	if cf.DI_SYLLABLE:
		return pair_params
	else:
		return pair_params[0]


def check_dup_fn(random_param, entire_param_sets):
	'''
	check if the newly random vocal tract parameter is duplicated with an
	existing random param
	'''
	# sort list
	random_param.sort()
	# group by item, if item in it list is duplicated, it will be group into (select only) one item.
	new_random_param = list(item for item,_ in itertools.groupby(random_param))
	# calculated the number of duplication item
	num_dup = len(random_param) - len(new_random_param)
	# check if the newly random params is duplicated with existing random params
	if entire_param_sets != []:
		for item in new_random_param:
			if item in entire_param_sets:
				new_random_param.remove(item)
				num_dup += 1
	
	return new_random_param, num_dup

def generate_vocaltract_parameter(data_size, predefine_params, entire_param_sets, is_check_dup = False):
	'''
	generate vocaltract parameter by randomizing
	'''
	gen_param_sets = []

	if is_check_dup:
		while(data_size > 0):
			random_param = [ random_fn(predefine_params) for i in range(data_size)]
			random_param, num_dup = check_dup_fn(random_param, entire_param_sets)
			data_size = num_dup
			gen_param_sets.extend(random_param)
			if num_dup > 0: print('Re-Generated %s data'%data_size)
	else:
		gen_param_sets.extend([ random_fn(predefine_params) for i in range(data_size)])
	
	return gen_param_sets

def repeat_label_by_n_speaker(gen_param_sets, n_speaker):
	'''
	To generate the same sound for different simulated speaker
	the parameter is being repeat by number of speaker 
	'''
	return [x for item in gen_param_sets for x in itertools.repeat(item, n_speaker)]

def scale_parameter(params, sid):
	'''
	Scale speaker vocaltract parameter of a syllable to the 
	scale of a new simulate speaker vocaltract parameter
	'''
	# scale to relative position
	scale_param = (params - param_low)/(param_high - param_low)
	# load min and max param of a simulated speaker
	sim_param = np.load(join(cf.SPKEAKER_SIM_DIR, 'speaker_param%s.npz'%sid))
	sim_high = sim_param['high']
	sim_low = sim_param['low']
	# scale back to the position of simulated speaker
	return scale_param*(sim_high - sim_low) + sim_low

def adjust_speaker_syllabel_parameter(gen_param_sets, speaker_sid):
	'''
	get speaker sid and generate param set to scale the vocaltract parameter
	to its simulated size
	'''
	return [scale_parameter(params, speaker_sid[idx]) for idx, params in enumerate(gen_param_sets)]
	

def get_speaker_sid(gen_param_sets, n_speaker):
	'''
	get a list speaker simulation id for generated speaker file
	where the speaker is a simulated speaker in that id
	example: create a list for speaker sid [0,1,2,3,...,6,0,1,2,3,...]
	'''
	return [ n%n_speaker for n in range(len(gen_param_sets))]


def generate_speaker_file(gen_param_sets, speaker_sid, speaker_idx):
	'''
	Main function to generate a speaker file for each generated 
	vocaltract parameters
	'''

	# Check error
	if os.path.isfile(cf.TAIL_SPEAKER): 
		# Read speaker template
		speaker_tail = open(cf.TAIL_SPEAKER,'r').read()
	else:
		raise ValueError('Tail file not exist, %s'%cf.TAIL_SPEAKER)

	# Create speaker folder
	os.makedirs(join(cf.DATASET_DIR, 'speaker'), exist_ok = True)
	# generate list of speaker filename
	speaker_filenames = ["speaker%s.speaker"%str(n+speaker_idx) for n, _ in enumerate(gen_param_sets)]
	# keep tract of speaker id
	speaker_idx += len(gen_param_sets)

	for idx, params in enumerate(gen_param_sets):
		# path to speaker head file
		speaker_head_file = join(cf.SPKEAKER_SIM_DIR, 'speaker_s%s.speaker'%speaker_sid[idx])

		# check if speaker head file is exist
		if os.path.isfile(speaker_head_file): 
			speaker_head = open(speaker_head_file,'r').read()
		else: 
			raise ValueError('Head file not exist, %s'%speaker_head_file)
		ss.create_speaker(join(cf.DATASET_DIR, 'speaker',speaker_filenames[idx]), params, speaker_idx, speaker_head, speaker_tail, is_disyllable = cf.DI_SYLLABLE)
	
	return speaker_filenames, speaker_idx

def generate_gesture_file(n_ges, ges_idx):
	'''
	Main function to generate a gesture file for each generated 
	vocaltract parameters
	'''
	os.makedirs(join(cf.DATASET_DIR, 'ges'), exist_ok = True)
	# generate list of gesture filename
	ges_filenames = ["ges%s.ges"%str(x+ges_idx) for x in range(n_ges)]
	# keep tract of ges id
	ges_idx += n_ges
	for file in ges_filenames:
		ss.create_ges(join(cf.DATASET_DIR, 'ges',file), cf.DI_SYLLABLE)
	return ges_filenames, ges_idx

def ges_to_wav(output_file_set, speaker_file_list, gesture_file_list, feedback_file_list, output, job_num):
	'''
	generate wave file from a given generated gesture and speaker
	'''
	# initialize vocaltractlab application
	VTL = initiate_VTL()
	# compute in c 
	for i, output_file in enumerate(output_file_set):
		speaker_file_name = ctypes.c_char_p(str.encode(speaker_file_list[i]))
		gesture_file_name = ctypes.c_char_p(str.encode(gesture_file_list[i]))
		wav_file_name = ctypes.c_char_p(str.encode(output_file))
		feedback_file_name = ctypes.c_char_p(str.encode(feedback_file_list[i]))
		# generated wave
		failure = VTL.vtlGesToWav(speaker_file_name,  # input
								  gesture_file_name,  # input
								  wav_file_name,  # output
								  feedback_file_name)  # output
	# keep tract of any failure
	output.put(failure)

def generate_sound(speaker_filenames, ges_filenames, sound_idx):
	'''
	Main function to generate a audio wave for each generated 
	vocaltract parameters
	'''

	# Create output folder
	os.makedirs(join(cf.DATASET_DIR, 'sound'), exist_ok = True)
	os.makedirs(join(cf.DATASET_DIR, 'feedback'), exist_ok = True)
	# generate list of audio filename
	sound_sets = ["sound%s.wav"%str(x+sound_idx) for x,_ in enumerate(speaker_filenames)]
	# generate list of feedback from VTL filename
	feedback_filenames = ["feedback%s.txt"%str(x+sound_idx) for x,_ in enumerate(speaker_filenames)]
	# keep tract of audio id
	sound_idx += len(speaker_filenames)

	# create a list of file path
	output_file_set = [join(cf.DATASET_DIR, 'sound', sound) for sound in sound_sets]
	speaker_file_set = [join(cf.DATASET_DIR, 'speaker', file) for file in speaker_filenames]
	ges_file_set = [join(cf.DATASET_DIR, 'ges', file) for file in ges_filenames]
	feedback_file_set = [join(cf.DATASET_DIR, 'feedback', file) for file in feedback_filenames]

	# Start multiprocess
	# Define an output queue
	output = mp.Queue()

	processes = []
	for i in range(cf.NJOB):
		start = i*int(len(sound_sets)/cf.NJOB)
		end = (i+1)*int(len(sound_sets)/cf.NJOB) if i != cf.NJOB-1 else len(sound_sets)
		# Setup a list of processes that we want to run
		processes.append(mp.Process(target=ges_to_wav, args=(output_file_set[start:end], speaker_file_set[start:end], ges_file_set[start:end], feedback_file_set[start:end],output, i)))

	# Run processes
	for p in processes:
		p.start()

	# Exit the completed processes
	for p in processes:
		p.join()

	failures = [output.get() for p in processes]
	if any(failures) != 0: raise ValueError('Error at file: ',failures)
	return sound_sets, sound_idx

def load_audio_from_list(sound_sets, sample_rate, parent_dir=''):
	audio_paths = [join(parent_dir, file) for file in sound_sets]
	return [ librosa.load(file, sr=sample_rate)[0] for file in audio_paths ]

def is_nonsilent(audio_data, threshold=0.8):
	'''
	Çheck if the sound is silent sound.
	return a list of index of non silent audio sound. 
	return in list format
	'''
	# if audio consist mostly non zero, indicating non silent sound
	return [idx for idx,data in enumerate(audio_data) if (np.count_nonzero(data) > threshold*data.shape[0])]

def filter_silent_sound(audio_data, sound_sets, gen_param_sets, speaker_sid):
	'''
	Main function to filter the silent sound.
	This function return list index of of non_silent sound 
	'''
	idx_list = is_nonsilent(audio_data)
	# convert back to list for consistancy purpose
	non_silent_sound_sets = np.array(sound_sets)[idx_list].tolist() 
	non_silent_param_sets = np.array(gen_param_sets)[idx_list].tolist() 
	# filter speaker sid for later used (in scale function)
	non_silent_sid = np.array(speaker_sid)[idx_list].tolist()
	# count silent sound
	silent_count = len(sound_sets) - len(idx_list)
	
	return non_silent_sound_sets, non_silent_param_sets, non_silent_sid, silent_count

def save_state(speaker_idx, ges_idx, sound_idx, total_speaker_sid, 
	entire_sound_sets, entire_param_sets, 
	ns_sound_sets, ns_param_sets, ns_sid):
	'''
	Save state for continuous data generation
	'''
	# create data folder
	os.makedirs(join(cf.DATASET_DIR, 'npy'), exist_ok=True)
	np.savez(join(cf.DATASET_DIR, 'npy','dataset.npz'), 
		state = np.array([speaker_idx, ges_idx, sound_idx]),
		total_speaker_sid=np.array(total_speaker_sid), 
		entire_sound_sets=np.array(entire_sound_sets),
		entire_param_sets = np.array(entire_param_sets),
		ns_sound_sets = np.array(ns_sound_sets),
		ns_param_sets = np.array(ns_param_sets),
		ns_sid = np.array(ns_sid))

def load_state():
	'''
	load state for continuous data generation
	'''
	try:
		data = np.load(join(cf.DATASET_DIR, 'npy','dataset.npz'))
		state = data['state'].tolist()
		speaker_idx, ges_idx, sound_idx  = state[0], state[1], state[2]

		total_speaker_sid = data['total_speaker_sid'].tolist()
		entire_sound_sets = data['entire_sound_sets'].tolist()
		entire_param_sets = data['entire_param_sets'].tolist()
		ns_sound_sets = data['ns_sound_sets'].tolist()
		ns_param_sets = data['ns_param_sets'].tolist()
		ns_sid = data['ns_sid'].tolist()
		
	except:
		speaker_idx = 0
		ges_idx = 0
		sound_idx = 0
		total_speaker_sid = []
		entire_sound_sets = []
		entire_param_sets = []
		ns_sound_sets = []
		ns_param_sets = []
		total_block_sound = []
		total_block_label = []
		ns_sid = []
		
	return speaker_idx, ges_idx, sound_idx, total_speaker_sid, entire_sound_sets, entire_param_sets, ns_sound_sets, ns_param_sets, ns_sid

def reset(*args):
	'''
	reset state if not continue
	'''
	try: 
		for folder in args:
			shutil.rmtree(folder)
	except:
		print('Folder Already Empty')

def clean_folder():

	try: 
		for folder in ['speaker', 'ges', 'feedback', 'speaker_sim']:
			shutil.rmtree(join(cf.DATASET_DIR, folder))
	except:
		print('Folder Already Empty')

def main():
	'''
	main function to run data generator
	'''
	# reset state if not continue generating data.
	if not cf.CONT:
		print('[INFO] Reset state')
		reset(cf.DATASET_DIR)
	else:
		print('[INFO] Continue generating data')

	# simulated speaker vocaltract from given scale list.
	for sid, scale in enumerate(cf.SPEAKER_N):
		ss.simulate_speaker(scale, sid)

	global_time = time()

	# load predefine parameter
	predefine_params = load_file_csv(cf.PREDEFINE_PARAM_FILE)
	counter = 1

	while(counter <= cf.N_SPLIT):

		print('\n\n------------------------------------------')
		print('Step %s/%d'%(counter,cf.N_SPLIT))
		print('------------------------------------------\n\n')
		
		# load state if exist
		print('[INFO] Loading program states')
		speaker_idx, ges_idx, sound_idx, total_speaker_sid, entire_sound_sets, entire_param_sets, ns_sound_sets, ns_param_sets, ns_sid = load_state()
		
		# main generator algorithm
		print('[INFO] Generating random parameters')
		gen_param_sets = generate_vocaltract_parameter(int(cf.DATASIZE/cf.N_SPLIT), predefine_params, entire_param_sets)	
		print('[INFO] Generating list of speaker id')
		speaker_sid = get_speaker_sid(gen_param_sets, n_speaker=len(cf.SPEAKER_N))
		print('[INFO] Adjusting vocaltract parameters')
		gen_param_sets = adjust_speaker_syllabel_parameter(gen_param_sets, speaker_sid)
		print('[INFO] Generating speaker file')
		speaker_filenames, speaker_idx = generate_speaker_file(gen_param_sets, speaker_sid, speaker_idx)
		print('[INFO] Generating ges file')
		ges_filenames, ges_idx = generate_gesture_file(len(gen_param_sets), ges_idx)
		print('[INFO] Generating sound')
		sound_sets, sound_idx = generate_sound(speaker_filenames, ges_filenames, sound_idx)
		print('[INFO] Loading audio data')
		audio_data = load_audio_from_list(sound_sets, sample_rate=cf.SOUND_SAMPLING_RATE, parent_dir=join(cf.DATASET_DIR, 'sound'))
		print('[INFO] Filtering silent audio')
		non_silent_sound_sets, non_silent_param_sets, non_silent_sid, silent_count = filter_silent_sound(audio_data, sound_sets,gen_param_sets, speaker_sid)
		
		print('[INFO] Saving state')
		# store data in main list
		entire_param_sets += gen_param_sets
		total_speaker_sid += speaker_sid
		entire_sound_sets += sound_sets
		
		ns_sound_sets += non_silent_sound_sets
		ns_param_sets += non_silent_param_sets
		ns_sid += non_silent_sid
 
 		# save data
		save_state(speaker_idx, ges_idx, sound_idx, 
			total_speaker_sid, 
			entire_sound_sets, 
			entire_param_sets, 
			ns_sound_sets, 
			ns_param_sets,
			ns_sid)
		
		print('\n\n------------------------------------------')
		print('Successfully export data')
		print('Block sound found: ', silent_count)
		print('End of step %s/%d'%(counter,cf.N_SPLIT))
		print('------------------------------------------\n\n')
		counter += 1

	# Successfully generate

	if cf.CLEAN_FILE:
		print('[INFO] Cleansing')
		clean_folder()


	print('Successfully generated audio data')
	print('[Time: %.3fs]'%(time()-global_time))

	# Log report
	if os.path.isfile(cf.DATA_LOG_FILE):
		log = open(cf.DATA_LOG_FILE, 'a')
	else:
		log = open(cf.DATA_LOG_FILE, 'w')
		
	log.write('\n------------------------------------------\n')
	log.write('DATASET: %s\n'%cf.DATASET_NAME)
	if not cf.CONT: log.write('Generated Date: %s\n'%datetime.now().strftime("%Y %B %d %H:%M"))
	if cf.CONT: log.write('Modified Date: %s\n'%datetime.now().strftime("%Y %B %d %H:%M"))
	log.write('Data Description: %s\n'%cf.DATA_DESCRIPTION)
	log.write('Di Syllable Data: %s\n'%str(cf.DI_SYLLABLE))
	log.write('Dataset Directory: %s\n'%cf.DATASET_DIR)
	log.write('------------------------------------------\n')
	log.write('CONFIGURATIONS\n')
	log.write('Data size: %s\n'%cf.DATASIZE)
	log.write('Sampling step: %s\n'%cf.SAMPLING_STEP)
	log.write('Default parameter used: %s\n'%cf.PREDEFINE_PARAM_FILE)
	log.write('Simulated speaker scale: %s\n'%cf.SPEAKER_N)
	log.write('Sound sampling rate: %s\n'%cf.SOUND_SAMPLING_RATE)
	log.write('------------------------------------------\n')
	log.write('Entire param set: %s\n'%str(np.array(entire_param_sets).shape))
	log.write('Entire sound set: %s\n'%str(np.array(entire_sound_sets).shape))
	log.write('Non silent audio: %s\n'%str(np.array(ns_sound_sets).shape))
	log.write('Non silent labels: %s\n'%str(np.array(ns_param_sets).shape))
	log.write('\n------------------------------------------\n')

if __name__ == '__main__':
	main()

	