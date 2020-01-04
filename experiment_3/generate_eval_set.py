'''
This file convert the given vocaltract parameter (csv)
to a audio using VTL
'''
import numpy as np
import pandas as pd
from os import makedirs
from os.path import join, exists
import lib.dev_gen as gen
import argparse
import config as cf

def main(args):	
	# check if file exist
	if not exists(args.data_file):
		raise ValueError('File %s not found'%args.data_file)
	# read dataframe and covert to numpy
	df = pd.read_csv(args.data_file)
	params = df.loc[:, df.columns != 'syllable'].values
	syllable_name = df['syllable'].values
	# create output path
	makedirs(args.output_path, exist_ok=True)
	# if creating disyllable, joining the mono-syllabic together
	if args.disyllable:
		params, syllable_name = gen.create_disyllabic_parameter(params, syllable_name)
	# save to npy format
	np.save(file=join(args.output_path, 'syllable_name.npy'), arr=syllable_name)
	# convert from param to wav
	gen.convert_param_to_wav(params, args.output_path, args.disyllable)
	
if __name__ == '__main__':
	parser = argparse.ArgumentParser("Predicting vocaltract from audio")
	parser.add_argument("data_file", help="data file, consisting of parameter and syllable name", type=str)
	parser.add_argument("output_path", help="output directory", type=str)
	parser.add_argument('--mono', dest='disyllable', help="is data disyllable or monosyllable ['mono','di']", action='store_false')
	parser.add_argument('--di', dest='disyllable', help="is data disyllable or monosyllable ['mono','di']", action='store_true')
	args = parser.parse_args()
	main(args)