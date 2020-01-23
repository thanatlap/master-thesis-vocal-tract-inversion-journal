'''
This script extract signal and save as csv
This script use function from preprocess
'''

import numpy as np
import librosa 
from librosa import feature
import matplotlib.pyplot as plt
import os
import math
from os.path import join
import random
import itertools
import argparse
import preprocess as prep

def main(args):
	# check if data path is existed or not
	if not os.path.exists(args.data_path):
		raise ValueError('[ERROR] Data path %s is not exist'%args.data_path)

	# create output file
	os.makedirs(args.output_path, exist_ok=True)
	# import data, note that if mode=predict, labels is [].
	audio_paths, labels = prep.import_data(args.data_path, mode=args.mode)
	audio_data = prep.load_audio(audio_paths, args.sample_rate)
	# if not predict data, the label is given and need to be scaled
	if args.mode != 'predict':
		labels = prep.scale_speaker_syllable(labels, args.data_path, mode=args.mode)
	# export signal to numpy
	np.savez(join(args.output_path,'raw_signal_data.npz'),
			labels= labels,
			raw_signal = audio_data)
	print('[INFO] Export signal data to numpy successful')

if __name__ == '__main__':
	parser = argparse.ArgumentParser("Extract signal to numpy format")
	parser.add_argument("data_path", help="data parent directory", type=str)
	parser.add_argument("output_path", help="output directory", type=str)
	parser.add_argument("mode", help="data mode ['training', 'eval', 'predict']", type=str)
	parser.add_argument("--sample_rate", help="audio sample rate", type=int, default=16000)
	args = parser.parse_args()
	main(args)

