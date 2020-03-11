import ctypes, os, sys, shutil, math, itertools, librosa, csv
from os.path import join, exists
from os import makedirs
import pandas as pd
import numpy as np
import multiprocessing as mp
from shutil import copyfile

import simulate_speaker as ss

def load_file_csv(file):
	if exists(file):
		df = pd.read_csv(file)
		return df
	else:
		raise ValueError('File %s not found!'%file)

def transform_param_data_to_npy(param_sets, param_col = 'PARAM'):

	syllable_labels = param_sets[param_col].values
	param_sets = param_sets.drop([param_col], axis=1)
	syllable_params = param_sets.values
	# param_names = param_sets.columns.values
	return syllable_params, syllable_labels

def load_predefined_adult(file):
	if exists(file):
		predefined = np.load(file)
		adult_high = predefined['adult_high']
		adult_low = predefined['adult_low']
		return adult_high, adult_low
	else:
		raise ValueError('File %s not found!'%file)

def load_predefined_all(file):
	if exists(file):
		predefined = np.load(file)
		adult_high = predefined['adult_high']
		adult_low = predefined['adult_low']
		adult_neutral = predefined['adult_neutral']
		infant_high = predefined['infant_high']
		infant_low = predefined['infant_low']
		infant_neutral = predefined['infant_neutral']
		adult_narrow = predefined['adult_narrow']
		infant_narrow = predefined['infant_narrow']
		adult_wide = predefined['adult_wide']
		infant_wide = predefined['infant_wide']
		param_name = predefined['param_name']
		return adult_high, adult_low
	else:
		raise ValueError('File %s not found!'%file)

	

def rescale_parameter(batch_gen_param, speaker_sid, current_high, current_low, sim_path):
	'''
	get speaker sid and generate param set to scale the vocaltract parameter
	to its simulated size
	'''
	def _rescale_fn(params, sid):
		'''
		Scale speaker vocaltract parameter of a syllable to the 
		scale of a new simulate speaker vocaltract parameter
		'''
		# scale to relative position
		scale_param = (params - current_low)/(current_high - current_low)
		# load min and max param of a simulated speaker
		sim_param = np.load(join(sim_path, 'speaker_param_s%s.npz'%sid))
		sim_high = sim_param['high']
		sim_low = sim_param['low']
		# scale back to the position of simulated speaker
		return (scale_param*(sim_high - sim_low) + sim_low).tolist()

	return [_rescale_fn(params, speaker_sid[idx]) for idx, params in enumerate(batch_gen_param)]

def get_speaker_sid(batch_gen_param, n_speaker):
	'''
	get a list speaker simulation id for generated speaker file
	where the speaker is a simulated speaker in that id
	example: create a list for speaker sid [0,1,2,3,...,6,0,1,2,3,...]
	'''
	return [ n%n_speaker for n in range(len(batch_gen_param))]

def generate_speaker_file(batch_params, speaker_sid, speaker_idx,
	is_disyllable,
	speaker_tail_path,
	output_parent_dir):
	'''
	Main function to generate a speaker file for each generated 
	vocaltract parameters
	'''

	# Check error
	if os.path.isfile(speaker_tail_path): 
		# Read speaker template
		speaker_tail = open(speaker_tail_path,'r').read()
	else:
		raise ValueError('Tail file not exist, %s'%speaker_tail_path)

	# Create speaker folder
	makedirs(join(output_parent_dir, 'speaker'), exist_ok = True)
	# generate list of speaker filename
	speaker_filenames = ["speaker%s.speaker"%str(n+speaker_idx) for n, _ in enumerate(batch_params)]
	# keep tract of speaker id
	speaker_idx += len(batch_params)

	for idx, params in enumerate(batch_params):
		# path to speaker head file
		speaker_head_file = join(output_parent_dir,'simulated_speakers', 'speaker_s%s_partial.speaker'%speaker_sid[idx])

		# check if speaker head file is exist
		if os.path.isfile(speaker_head_file): 
			speaker_head = open(speaker_head_file,'r').read()
		else: 
			raise ValueError('Head file not exist, %s'%speaker_head_file)
		ss.create_speaker(join(output_parent_dir, 'speaker',speaker_filenames[idx]), params, speaker_head, speaker_tail, is_disyllable)
	
	return speaker_filenames, speaker_idx

def generate_gesture_file(n_ges, ges_idx, output_parent_dir,
	ges_head_file,
	is_disyllable):
	'''
	Main function to generate a gesture file for each generated 
	vocaltract parameters
	'''
	makedirs(join(output_parent_dir, 'ges'), exist_ok = True)
	# generate list of gesture filename
	ges_filenames = ["ges%s.ges"%str(x+ges_idx) for x in range(n_ges)]
	# keep tract of ges id
	ges_idx += n_ges
	for file in ges_filenames:
		ss.create_ges(join(output_parent_dir, 'ges',file), is_disyllable, ges_head_file)
	return ges_filenames, ges_idx

def initiate_VTL(vtl_file_path):
	if exists(vtl_file_path):
		# Call VTL application
		VTL = ctypes.cdll.LoadLibrary(vtl_file_path)
	else:
		raise ValueError('Path %s does not exist'%vtl_file_path)
	return VTL

def ges_to_wav(output_file_set, speaker_file_list, gesture_file_list, feedback_file_list, output, job_num, vtl_file_path):
	'''
	generate wave file from a given generated gesture and speaker
	'''
	# initialize vocaltractlab application
	VTL = initiate_VTL(vtl_file_path)
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

def generate_sound(speaker_filenames, ges_filenames, sound_idx, vtl_file_path, output_parent_dir, njob):
	'''
	Main function to generate a audio wave for each generated 
	vocaltract parameters
	'''

	# Create output folder
	makedirs(join(output_parent_dir, 'sound'), exist_ok = True)
	makedirs(join(output_parent_dir, 'feedback'), exist_ok = True)
	# generate list of audio filename
	sound_sets = ["sound%s.wav"%str(x+sound_idx) for x,_ in enumerate(speaker_filenames)]
	# generate list of feedback from VTL filename
	feedback_filenames = ["feedback%s.txt"%str(x+sound_idx) for x,_ in enumerate(speaker_filenames)]
	# keep tract of audio id
	sound_idx += len(speaker_filenames)

	# create a list of file path
	output_file_set = [join(output_parent_dir, 'sound', sound) for sound in sound_sets]
	speaker_file_set = [join(output_parent_dir, 'speaker', file) for file in speaker_filenames]
	ges_file_set = [join(output_parent_dir, 'ges', file) for file in ges_filenames]
	feedback_file_set = [join(output_parent_dir, 'feedback', file) for file in feedback_filenames]

	# Start multiprocess
	# Define an output queue
	output = mp.Queue()

	processes = []
	for i in range(njob):
		start = i*int(len(sound_sets)/njob)
		end = (i+1)*int(len(sound_sets)/njob) if i != njob-1 else len(sound_sets)
		# Setup a list of processes that we want to run
		processes.append(mp.Process(target=ges_to_wav, args=(output_file_set[start:end], speaker_file_set[start:end], ges_file_set[start:end], feedback_file_set[start:end],output, i, vtl_file_path)))

	# Run processes
	for p in processes:
		p.start()

	# Exit the completed processes
	for p in processes:
		p.join()

	failures = [output.get() for p in processes]
	if any(failures) != 0: raise ValueError('Error at file: ',failures)
	return sound_sets, sound_idx

def load_audio_from_list(file_set, sample_rate):
	return [librosa.load(file, sr=sample_rate)[0] for file in file_set]

def is_nonsilent(audio_data, threshold):
	'''
	Çheck if the sound is silent sound. return a list of index of non silent audio sound.
	threshold is constant at 0.9
	'''
	# if audio consist mostly non zero, indicating non silent sound
	return [idx for idx,data in enumerate(audio_data) if (np.count_nonzero(data) > threshold*data.shape[0])]
	

def filter_nonsound(sound_sets, total_aggregate_param, total_speaker_sid, total_phonetic, sample_rate, njob, output_parent_dir, threshold=0.9):

	file_set = [join(output_parent_dir, 'sound', file) for file in sound_sets]

	audio_data = load_audio_from_list(file_set, sample_rate)
	idx_list = is_nonsilent(audio_data, threshold)
	batch_ns_audio = np.array(audio_data)[idx_list].tolist()
	batch_ns_param = np.array(total_aggregate_param)[idx_list].tolist()
	batch_ns_sid = np.array(total_speaker_sid)[idx_list].tolist()
	batch_ns_phonetic = np.array(total_phonetic)[idx_list].tolist()
	# count silent sound
	silent_count = len(audio_data) - len(idx_list)
	return batch_ns_audio, batch_ns_param, batch_ns_sid, batch_ns_phonetic, silent_count

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

def create_speaker_from_template(syllable_params, param_names, output_path, speaker_header_file, speaker_tail_file, is_disyllable = False):

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

def create_ges_from_template(syllable_params, output_path, is_disyllable):
	
	ges_path = join(output_path,'ges')
	os.makedirs(ges_path, exist_ok=True)

	ges_file = 'gesture_disyllable_template.ges' if is_disyllable else 'gesture_monosyllable_template.ges'
	ges_filenames = [ges_file]*len(syllable_params)
	shutil.copy('templates/'+ges_file, ges_path)

	return ges_filenames