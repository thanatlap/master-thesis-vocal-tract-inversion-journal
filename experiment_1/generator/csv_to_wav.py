'''
This file convert the given vocaltract parameter (csv)
to a audio using VTL
'''
import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists
import gen_utils as gen
import argparse
import csv
from time import time

def param_to_wav(syllable_params, param_names, output_path, speaker_header_file, speaker_tail_file, is_disyllable):

	start = time()
	speaker_filenames = gen.create_speaker_from_template(syllable_params=syllable_params, 
		param_names=param_names, 
		output_path=output_path, 
		speaker_header_file=speaker_header_file, 
		speaker_tail_file =speaker_tail_file, 
		is_disyllable = is_disyllable)
	ges_filenames = gen.create_ges_from_template(syllable_params, output_path, is_disyllable=is_disyllable)
	sound_sets = gen.generate_sound(speaker_filenames, ges_filenames, 0, 'assets/VTL/VocalTractLabApi.dll', output_path, njob=4)
	print('Successfully convert label to sound [Time: %.3fs]'%(time()-start))
	return sound_sets

def main(args):	

	# check error and create output path
	if not exists(args.csv_path):
		raise ValueError('[ERROR] CSV file %s does not exist'%args.csv_path)
	if not exists(args.speaker_template_head):
		raise ValueError('[ERROR] Speaker header template %s does not exist'%args.speaker_template_head)
	if not exists(args.speaker_template_tail):
		raise ValueError('[ERROR] Speaker tail template %s does not exist'%args.speaker_template_tail)
	if args.syllable.lower() not in ['mono','di']:
		raise ValueError('[ERROR] Syllable mode %s is not match [mono, di]'%args.syllable)
	else:
		disyllable = True if args.syllable.lower() == 'di' else False
	if args.output_path == None:
		# the default outputpath is set for an acoustic evaluation dataset
		output_path = '../../data/d_eval' if disyllable else '../../data/m_eval'
	else:
		output_path = join('..','data', args.output_path)
	makedirs(output_path, exist_ok=True)

	# import csv data
	syllable_labels, syllable_params, param_names = gen.import_data_from_csv(args.csv_path)	
	# combine parameter to create disyllable
	if disyllable:
		syllable_params, syllable_labels = gen.create_disyllabic_parameter(syllable_params, syllable_labels) 

	gen.export_syllable_name(syllable_labels, output_path)
	# convert from param to wav
	sound_sets = param_to_wav(syllable_params, param_names, output_path, args.speaker_template_head, args.speaker_template_tail, disyllable)
	# export data
	np.savez(join(output_path,'csv_dataset.npz'), 
		syllable_params = syllable_params,
		syllable_labels = syllable_labels,
		sound_sets = sound_sets)

if __name__ == '__main__':
	parser = argparse.ArgumentParser("CSV to wav file")
	parser.add_argument("csv_path", help="csv file path", type=str)
	parser.add_argument("syllable", help="is data disyllable or monosyllable ['mono','di']", type=str)
	parser.add_argument('--head_filepath', dest='speaker_template_head', default='templates/speaker_head.speaker', help='speaker head template file path', type=str)
	parser.add_argument('--tail_filepath', dest='speaker_template_tail', default='templates/speaker_tail.txt', help='speaker tail template file path', type=str)

	parser.add_argument("--output_path", help="output directory", default=None, type=str)
	args = parser.parse_args()
	main(args)