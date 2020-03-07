import ctypes, sys, shutil, math, itertools, librosa
from os.path import join, exists, isfile
from os import makedirs
import numpy as np
import pandas as pd
from time import time
from datetime import datetime
import multiprocessing as mp
from functools import partial

import config as cf
import gen_tools as gen
import simulate_speaker as ss
import random_param as rparam 

# fix error from numpy where np load allow_pickle set to False as default
np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# --- Variable ---

PARAM = 'PARAM'

def generate_vocaltract_parameter(batch_size, predefined_syllables, total_aggregate_param):

	batch_gen_param = []

	while(batch_size > 0):
		random_param = [ rparam.randomly_select_syllable_pair(predefined_syllables, cf.DI_SYLLABLE, cf.MIN_MAX_PERCENT_CHANGE) for i in range(batch_size)]
		print('[INFO] Check Duplicated Item')
		random_param, batch_size = rparam.check_duplicate_and_remove(random_param, total_aggregate_param)
		batch_gen_param.extend(random_param)
		if batch_size > 0: print('[INFO] Re-Generated %s data'%batch_size)

	return batch_gen_param

def save_state(speaker_idx, ges_idx, sound_idx, total_speaker_sid, 
	total_aggregate_sound, total_aggregate_param, prev_export_status):
	'''
	Save state for continuous data generation
	'''
	# create data folder
	makedirs(join(cf.DATASET_DIR, 'vars'), exist_ok=True)
	np.savez(join(cf.DATASET_DIR, 'vars','data_gen_state.npz'), 
		state = np.array([speaker_idx, ges_idx, sound_idx, prev_export_status]),
		total_speaker_sid=np.array(total_speaker_sid), 
		total_aggregate_sound=np.array(total_aggregate_sound),
		total_aggregate_param = np.array(total_aggregate_param))

def load_state():
	'''
	load state for continuous data generation
	'''
	try:
		data = np.load(join(cf.DATASET_DIR, 'vars','data_gen_state.npz'))
		state = data['state'].tolist()
		speaker_idx, ges_idx, sound_idx, prev_export_status  = state[0], state[1], state[2], state[3]

		total_speaker_sid = data['total_speaker_sid'].tolist()
		total_aggregate_sound = data['total_aggregate_sound'].tolist()
		total_aggregate_param = data['total_aggregate_param'].tolist()
		print('[INFO] Load from previous state')
		
	except:
		print('[INFO] State not found, initialize new variable')
		speaker_idx = 0
		ges_idx = 0
		sound_idx = 0
		prev_export_status = 0
		total_speaker_sid = []
		total_aggregate_sound = []
		total_aggregate_param = []
		
	return speaker_idx, ges_idx, sound_idx, total_speaker_sid, total_aggregate_sound, total_aggregate_param, prev_export_status

def export_dataset(ns_audio_data, ns_aggregate_param, ns_sid):

	np.savez(join(cf.DATASET_DIR,'dataset.npz'), 
		ns_audio_data=np.array(ns_audio_data), 
		ns_aggregate_param=np.array(ns_aggregate_param),
		ns_sid = np.array(ns_sid))

def import_dataset():

	try:
		data = np.load(join(cf.DATASET_DIR,'dataset.npz'))
		ns_audio_data = data['ns_audio_data'].tolist()
		ns_aggregate_param = data['ns_aggregate_param'].tolist()
		ns_sid = data['ns_sid'].tolist()
	except:
		ns_audio_data = []
		ns_aggregate_param = []
		ns_sid = []
	return ns_audio_data, ns_aggregate_param, ns_sid

def reset(*args):
	try: 
		for folder in args:
			shutil.rmtree(folder)
	except Exception as e: 
		print(e)
		print('[INFO] Folder Already Empty')

def clean_folder(*args):

	try: 
		for folder in args:
			shutil.rmtree(join(cf.DATASET_DIR, folder))
	except:
		print('[INFO] Folder Already Empty')

def clean_up_file():
	if cf.CLEAN_FILE:
		clean_folder( 'feedback')
	if cf.CLEAN_SOUND:
		clean_folder('speaker', 'ges', 'sound')

def check_file_exist(*args):

	for file in args:
		if not exists(file):
			raise ValueError('[ERROR] File %s does not exist'%file)

def check_is_continue_or_replace():

	if cf.CONT:
		print('[INFO] Continue generating data')
		if not exists(join(cf.DATASET_DIR, 'dataset.npz')):
			print('[WARNING] dataset.npz is not found!')
	elif not cf.CONT and cf.REPLACE_FOLDER:
		print('[INFO] Reset state')
		print(cf.DATASET_DIR)
		reset(cf.DATASET_DIR)
	else:
		index = 1
		dataset_dir = cf.DATASET_DIR
		while True:
			if exists(dataset_dir):
				dataset_dir = cf.DATASET_DIR+'_%s'%index
				index += 1
			else:
				cf.DATASET_DIR = dataset_dir
				break
		makedirs(cf.DATASET_DIR)

def main():
	# check for error
	print('[INFO] Check required file')
	check_file_exist(cf.VTL_FILE, cf.PREDEFINE_PARAM_FILE, cf.ADULT_SPEAKER_HEADER_FILE,
		cf.INFANT_SPEAKER_HEADER_FILE, cf.TAIL_SPEAKER, cf.GES_HEAD)
	# check if continue or replace the main output folder
	print('[INFO] Check continue or replace')
	check_is_continue_or_replace()

	# load predefine parameter
	print('[INFO] Load csv predefined param template')
	predefined_data = gen.load_file_csv(cf.PREDEFINE_PARAM_FILE)
	print('[INFO] Transform param to npy')
	predefined_syllables, syllable_labels = gen.transform_param_data_to_npy(predefined_data, param_col=PARAM)
	print('[INFO] Simulated speaker')
	ss.simulate_speaker_from_given_ratio(cf.SPEAKER_N, predefined_syllables, syllable_labels,
		adult_header_file=cf.ADULT_SPEAKER_HEADER_FILE,
		infant_header_file=cf.INFANT_SPEAKER_HEADER_FILE,
		speaker_tail_file=cf.TAIL_SPEAKER,
		dataset_folder=cf.DATASET_DIR)

	param_high, param_low = gen.load_predefined_adult(cf.PREDEFINED_PARAM)

	start_time = time()
	timestamp = datetime.now().strftime("%Y %B %d %H:%M")
	split_counter = 0

	# load state if exist
	print('[INFO] Loading program states')
	speaker_idx, ges_idx, sound_idx, total_speaker_sid, total_aggregate_sound, total_aggregate_param, prev_export_status = load_state()

	if prev_export_status == 1:
		prev_ns_aggregate_param = np.load(join(cf.DATASET_DIR,'dataset.npz'))['ns_aggregate_param'].tolist()
	else:
		prev_ns_aggregate_param = []

	while(split_counter < cf.N_SPLIT):
		split_counter += 1
		print('------------------------------------------')
		print('[INFO] Step %s/%d'%(split_counter,cf.N_SPLIT))		
		# main generator algorithm
		print('[INFO] Generating random parameters')
		batch_gen_param = generate_vocaltract_parameter(int(cf.DATASIZE/cf.N_SPLIT), predefined_syllables, total_aggregate_param+prev_ns_aggregate_param)	
		print('[INFO] Generating list of speaker id')
		speaker_sid = gen.get_speaker_sid(batch_gen_param, n_speaker=len(cf.SPEAKER_N))
		print('[INFO] Adjusting vocaltract parameters')
		batch_gen_param = gen.rescale_parameter(batch_gen_param, speaker_sid, param_high, param_low, cf.SIM_PATH)
		print('[INFO] Generating speaker file')
		speaker_filenames, speaker_idx = gen.generate_speaker_file(batch_gen_param, speaker_sid, speaker_idx, cf.DI_SYLLABLE, cf.TAIL_SPEAKER, cf.DATASET_DIR)
		print('[INFO] Generating ges file')
		ges_filenames, ges_idx = gen.generate_gesture_file(len(batch_gen_param), ges_idx, cf.DATASET_DIR, cf.GES_HEAD, cf.DI_SYLLABLE)
		print('[INFO] Generating sound')
		sound_sets, sound_idx = gen.generate_sound(speaker_filenames, ges_filenames, sound_idx, cf.VTL_FILE, cf.DATASET_DIR, cf.NJOB)
		
		# store data in main list
		total_aggregate_param += batch_gen_param
		total_speaker_sid += speaker_sid
		total_aggregate_sound += sound_sets
 
		# save data
		print('[INFO] Saving state')
		save_state(speaker_idx, 
			ges_idx, 
			sound_idx, 
			total_speaker_sid, 
			total_aggregate_sound, 
			total_aggregate_param,
			prev_export_status)
		
		print('[INFO] End of step %s/%d'%(split_counter,cf.N_SPLIT))
		
	print('[INFO] Loading audio data for filtering')
	batch_ns_audio, batch_ns_param, batch_ns_sid, silent_count = gen.filter_nonsound(total_aggregate_sound, total_aggregate_param, total_speaker_sid,
		cf.AUDIO_SAMPLE_RATE, cf.NJOB, cf.DATASET_DIR)
	
	if prev_export_status:
		ns_audio_data, ns_aggregate_param, ns_sid = import_dataset()
		ns_audio_data += batch_ns_audio
		ns_aggregate_param += batch_ns_param
		ns_sid += batch_ns_sid
	else:
		ns_audio_data = batch_ns_audio
		ns_aggregate_param = batch_ns_param
		ns_sid = batch_ns_sid

	print('[INFO] Export dataset')
	export_dataset(ns_audio_data, ns_aggregate_param, ns_sid)
	# total_* is not required to continue generated data
	save_state(speaker_idx,ges_idx,sound_idx,[],[],[], prev_export_status=1) 
	clean_up_file()
	print('[INFO] Successfully generated audio data')
	print('[INFO] Silent sound found: %s'%silent_count)
	total_time = time()-start_time
	print('[Time: %.3fs]'%total_time)

	# Log report
	log_filepath = join(cf.DATASET_DIR, 'data_description.txt')
	if isfile(log_filepath):
		log = open(log_filepath, 'a')
	else:
		log = open(log_filepath, 'w')
		
	log.write('\n========================================\n')
	log.write('DATASET: %s\n'%cf.DATASET_NAME)
	log.write('Generated Date: %s\n'%timestamp) if not cf.CONT else log.write('Modified Date: %s\n'%timestamp)
	log.write('Data Description: %s\n'%cf.DATA_DESCRIPTION)
	log.write('Disyllable Vowel: %s\n'%str(cf.DI_SYLLABLE))
	log.write('Dataset Directory: %s\n'%cf.DATASET_DIR)
	log.write('Total Generated Time: %s\n'%total_time)
	log.write('------------------------------------------\n')
	log.write('METADATA\n')
	log.write('Data size: %s\n'%cf.DATASIZE)
	log.write('Epoch/Split: %s/%s\n'%(split_counter,cf.N_SPLIT))
	log.write('Non silent param: %s\n'%str(np.array(ns_aggregate_param).shape))
	log.write('Non audio data: %s\n'%str(np.array(ns_audio_data).shape))
	log.write('Non speaker id: %s\n'%str(np.array(ns_sid).shape))
	log.write('Silent sound count: %s\n'%silent_count)
	log.write('------------------------------------------\n')
	log.write('HYPERPARAMETER\n')
	log.write('FILTER_THRES: %s\n'%cf.FILTER_THRES)
	log.write('SAMPLING_STEP: %s\n'%cf.SAMPLING_STEP)
	log.write('MIN_MAX_PERCENT_CHANGE: %s\n'%cf.MIN_MAX_PERCENT_CHANGE)
	log.write('RAMDOM_PARAM_NOISE_PROB: %s\n'%cf.RAMDOM_PARAM_NOISE_PROB)
	log.write('AUDIO_SAMPLE_RATE: %s\n'%cf.AUDIO_SAMPLE_RATE)
	log.write('PREDEFINE_PARAM_FILE: %s\n'%cf.PREDEFINE_PARAM_FILE)
	log.write('SPEAKER_N: %s\n'%cf.SPEAKER_N)
	log.write('GES_MIN_MAX_DURATION_DI: %s\n'%cf.GES_MIN_MAX_DURATION_DI)
	log.write('GES_MIN_MAX_DURATION_MONO: %s\n'%cf.GES_MIN_MAX_DURATION_MONO)
	log.write('GES_VARY_DURATION_DI: %s\n'%cf.GES_VARY_DURATION_DI)
	log.write('GES_TIME_CONST: %s\n'%cf.GES_TIME_CONST)
	log.write('GES_F0_INIT_MIN_MAX: %s\n'%cf.GES_F0_INIT_MIN_MAX)
	log.write('GES_F0_NEXT_MIN_MAX: %s\n'%cf.GES_F0_NEXT_MIN_MAX)
	log.write('------------------------------------------\n')
	log.write('CONFIGURATIONS\n')
	log.write('Replace Folder: %s\n'%cf.REPLACE_FOLDER)
	log.write('Clean Folder: %s\n'%cf.CLEAN_FILE)
	log.write('Clean Sound: %s\n'%cf.CLEAN_SOUND)
	log.write('NJOB: %s\n'%cf.NJOB)

if __name__ == '__main__':
	main()

