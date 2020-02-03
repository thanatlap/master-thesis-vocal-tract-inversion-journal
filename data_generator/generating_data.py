import ctypes, os, sys, shutil, math, itertools, librosa
from os.path import join, exists
from glob import glob
import numpy as np
import pandas as pd
from scipy import spatial
from scipy.spatial.distance import euclidean
from time import time
from datetime import datetime
import multiprocessing as mp
import config as cf
import simulate_speaker as ss
import random_param as rand_param 
from functools import partial

PARAM_HIGH = ss.adult_high
PARAM_LOW = ss.adult_low

PARAM = 'PARAM'

# fix error from numpy where np load allow_pickle set to False as default
np_load_old = partial(np.load)
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

def initiate_VTL():
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
		return pd.read_csv(file)
	else:
		raise ValueError('File %s not found!'%file)

def transform_param_data_to_npy(param_sets):

	syllable_labels = param_sets[PARAM].values
	param_sets = param_sets.drop([PARAM], axis=1)
	syllable_params = param_sets.values
	param_names = param_sets.columns.values
	return syllable_params, syllable_labels

def simulate_speaker_from_given_ratio(predefined_syllables, syllable_labels):
	'''
	 simulated speaker vocaltract from given scale list.
	 note that the first item on list must be 0.0
	'''
	for sid, scale in enumerate(cf.SPEAKER_N):
		ss.simulate_speaker(scale, sid, 
			adult_header_file=cf.ADULT_SPEAKER_HEADER_FILE, 
			infant_header_file=cf.INFANT_SPEAKER_HEADER_FILE, 
			speaker_tail_file=cf.TAIL_SPEAKER, 
			dataset_folder=cf.DATASET_DIR,
			predefined_syllables=predefined_syllables, 
			syllable_labels=syllable_labels)

def random_param_from_syllable_pair(from_idx, to_idx, predefined_syllables):
	'''
	add some noise from  randomize params function
	'''
	if np.random.uniform(0, high=1) < cf.RAMDOM_PARAM_NOISE_PROB:
		random_param = rand_param.randomize_noise_params(predefined_syllables, PARAM_HIGH, PARAM_LOW, sampling_step=cf.SAMPLING_STEP)
	else:
		random_param = rand_param.randomize_by_percent_change(predefined_syllables, from_idx, to_idx, cf.MIN_MAX_PERCENT_CHANGE[0], cf.MIN_MAX_PERCENT_CHANGE[1])
	return random_param

def randomly_select_syllable_pair(predefined_syllables):
	'''
	generate a vocal tract parameter set.
	'''
	# random select syllable pair  
	num_item_pair = 4 if cf.DI_SYLLABLE else 2
	while True:
		syllable_pair = [ np.random.randint(predefined_syllables.shape[0]) for i in range(num_item_pair)]
		# check duplicate index
		if ((len(syllable_pair) - len(set(syllable_pair))) == 0):
			break
	while True:
		random_syllables = [random_param_from_syllable_pair(syllable_pair[idx*2], syllable_pair[idx*2+1], predefined_syllables) for idx in range(int(num_item_pair/2))]
		# Check if the param is directly pick from default param since we will used this as a final testing set.
		if sum([0 if not (item in predefined_syllables.tolist()) else 1 for item in random_syllables]) == 0:
			break

	# If generate disyllable, return a pair of param else, return only first pair
	return random_syllables if cf.DI_SYLLABLE else random_syllables[0]

def check_duplicate_and_remove(random_param, total_aggregate_param):
	# sort list
	random_param.sort()
	# group by item, if item in it list is duplicated, it will be group into (select only) one item.
	new_random_param = list(item for item,_ in itertools.groupby(random_param))
	# calculated the number of duplication item
	duplicate_item_count = int(len(random_param) - len(new_random_param))
	# check if the newly random params is duplicated with existing random params
	if total_aggregate_param != []:
		for item in new_random_param:
			if item in total_aggregate_param:
				new_random_param.remove(item)
				duplicate_item_count += 1

	return new_random_param, duplicate_item_count

def generate_vocaltract_parameter(batch_size, predefined_syllables, total_aggregate_param):

	batch_gen_param = []

	while(batch_size > 0):
		random_param = [ randomly_select_syllable_pair(predefined_syllables) for i in range(batch_size)]
		print('[INFO] Check Duplicated Item')
		random_param, batch_size = check_duplicate_and_remove(random_param, total_aggregate_param)
		batch_gen_param.extend(random_param)
		if batch_size > 0: print('[INFO] Re-Generated %s data'%batch_size)

	return batch_gen_param

def scale_parameter(params, sid):
	'''
	Scale speaker vocaltract parameter of a syllable to the 
	scale of a new simulate speaker vocaltract parameter
	'''
	# scale to relative position
	scale_param = (params - PARAM_LOW)/(PARAM_HIGH - PARAM_LOW)
	# load min and max param of a simulated speaker
	sim_param = np.load(join(cf.DATASET_DIR, 'simulated_speakers', 'speaker_param_s%s.npz'%sid))
	sim_high = sim_param['high']
	sim_low = sim_param['low']
	# scale back to the position of simulated speaker
	return (scale_param*(sim_high - sim_low) + sim_low).tolist()

def adjust_speaker_syllabel_parameter(batch_gen_param, speaker_sid):
	'''
	get speaker sid and generate param set to scale the vocaltract parameter
	to its simulated size
	'''
	return [scale_parameter(params, speaker_sid[idx]) for idx, params in enumerate(batch_gen_param)]
	

def get_speaker_sid(batch_gen_param, n_speaker):
	'''
	get a list speaker simulation id for generated speaker file
	where the speaker is a simulated speaker in that id
	example: create a list for speaker sid [0,1,2,3,...,6,0,1,2,3,...]
	'''
	return [ n%n_speaker for n in range(len(batch_gen_param))]


def generate_speaker_file(batch_gen_param, speaker_sid, speaker_idx):
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
	speaker_filenames = ["speaker%s.speaker"%str(n+speaker_idx) for n, _ in enumerate(batch_gen_param)]
	# keep tract of speaker id
	speaker_idx += len(batch_gen_param)

	for idx, params in enumerate(batch_gen_param):
		# path to speaker head file
		speaker_head_file = join(cf.DATASET_DIR,'simulated_speakers', 'speaker_s%s_partial.speaker'%speaker_sid[idx])

		# check if speaker head file is exist
		if os.path.isfile(speaker_head_file): 
			speaker_head = open(speaker_head_file,'r').read()
		else: 
			raise ValueError('Head file not exist, %s'%speaker_head_file)
		ss.create_speaker(join(cf.DATASET_DIR, 'speaker',speaker_filenames[idx]), params, speaker_head, speaker_tail, is_disyllable = cf.DI_SYLLABLE)
	
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
		ss.create_ges(join(cf.DATASET_DIR, 'ges',file), cf.DI_SYLLABLE, cf.GES_HEAD)
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

def is_nonsilent(audio_data):
	'''
	Çheck if the sound is silent sound. return a list of index of non silent audio sound.
	'''
	# if audio consist mostly non zero, indicating non silent sound
	return [idx for idx,data in enumerate(audio_data) if (np.count_nonzero(data) > cf.FILTER_THRES*data.shape[0])]

def filter_silent_sound(audio_data, total_aggregate_param, total_speaker_sid):

	idx_list = is_nonsilent(audio_data)
	# convert to np array to use subset by list feature and back to list for consistency
	batch_ns_audio = np.array(audio_data)[idx_list].tolist()
	batch_ns_param = np.array(total_aggregate_param)[idx_list].tolist()
	batch_ns_sid = np.array(total_speaker_sid)[idx_list].tolist()
	# count silent sound
	silent_count = len(audio_data) - len(idx_list)
	
	return batch_ns_audio, batch_ns_param, batch_ns_sid, silent_count

def save_state(speaker_idx, ges_idx, sound_idx, total_speaker_sid, 
	total_aggregate_sound, total_aggregate_param, prev_export_status):
	'''
	Save state for continuous data generation
	'''
	# create data folder
	os.makedirs(join(cf.DATASET_DIR, 'vars'), exist_ok=True)
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
	except:
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
		if not os.path.exists(file):
			raise ValueError('[ERROR] File %s does not exist'%file)

def check_is_continue_or_replace():

	if cf.CONT:
		print('[INFO] Continue generating data')
		if not exists(cf.DATASET_DIR, 'dataset.npz'):
			print('[WARNING] dataset.npz is not found!')
	elif not cf.CONT and cf.REPLACE_FOLDER:
		print('[INFO] Reset state')
		reset(cf.DATASET_DIR)
	else:
		index = 1
		while exists(cf.DATASET_DIR):
			index += 1 # if folder already exist, add (2) at the end and so on
			cf.DATASET_DIR = cf.DATASET_DIR+'(%s)'%index
		os.makedirs(cf.DATASET_DIR)

def main():
	# check for error
	check_file_exist(cf.VTL_FILE, cf.PREDEFINE_PARAM_FILE, cf.ADULT_SPEAKER_HEADER_FILE,
		cf.INFANT_SPEAKER_HEADER_FILE, cf.TAIL_SPEAKER, cf.GES_HEAD)
	# check if continue or replace the main output folder
	check_is_continue_or_replace()

	# load predefine parameter
	predefined_data = load_file_csv(cf.PREDEFINE_PARAM_FILE)
	predefined_syllables, syllable_labels = transform_param_data_to_npy(predefined_data)
	simulate_speaker_from_given_ratio(predefined_syllables, syllable_labels)

	start_time = time()
	timestamp = datetime.now().strftime("%Y %B %d %H:%M")
	split_counter = 0

	# load state if exist
	print('[INFO] Loading program states')
	speaker_idx, ges_idx, sound_idx, total_speaker_sid, total_aggregate_sound, total_aggregate_param, prev_export_status = load_state()

	if prev_export_status == 1:
		total_aggregate_param += np.load(join(cf.DATASET_DIR,'dataset.npz'))['ns_aggregate_param'].tolist()

	while(split_counter < cf.N_SPLIT):
		split_counter += 1
		print('------------------------------------------')
		print('[INFO] Step %s/%d'%(split_counter,cf.N_SPLIT))		
		# main generator algorithm
		print('[INFO] Generating random parameters')
		batch_gen_param = generate_vocaltract_parameter(int(cf.DATASIZE/cf.N_SPLIT), predefined_syllables, total_aggregate_param)	
		print('[INFO] Generating list of speaker id')
		speaker_sid = get_speaker_sid(batch_gen_param, n_speaker=len(cf.SPEAKER_N))
		print('[INFO] Adjusting vocaltract parameters')
		batch_gen_param = adjust_speaker_syllabel_parameter(batch_gen_param, speaker_sid)
		print('[INFO] Generating speaker file')
		speaker_filenames, speaker_idx = generate_speaker_file(batch_gen_param, speaker_sid, speaker_idx)
		print('[INFO] Generating ges file')
		ges_filenames, ges_idx = generate_gesture_file(len(batch_gen_param), ges_idx)
		print('[INFO] Generating sound')
		sound_sets, sound_idx = generate_sound(speaker_filenames, ges_filenames, sound_idx)
		
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
			prev_export_status=0)
		
		print('[INFO] End of step %s/%d'%(split_counter,cf.N_SPLIT))
		
	print('[INFO] Loading audio data for filtering')
	audio_data = load_audio_from_list(total_aggregate_sound, sample_rate=cf.AUDIO_SAMPLE_RATE, parent_dir=join(cf.DATASET_DIR, 'sound'))
	print('[INFO] Filtering silent audio')
	batch_ns_audio, batch_ns_param, batch_ns_sid, silent_count = filter_silent_sound(audio_data, total_aggregate_param, total_speaker_sid)
	
	if cf.CONT:
		ns_audio_data, ns_aggregate_sound, ns_aggregate_param, ns_sid = import_dataset()
		ns_audio_data += batch_ns_audio
		ns_aggregate_param += batch_ns_param
		ns_sid += batch_ns_sid

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
	if os.path.isfile(log_filepath):
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
	log.write('Non silent param: %s\n'%str(ns_aggregate_param.shape))
	log.write('Non silent sound: %s\n'%str(ns_aggregate_sound.shape))
	log.write('Non audio data: %s\n'%str(ns_audio_data.shape))
	log.write('Non speaker id: %s\n'%str(ns_sid.shape))
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

	