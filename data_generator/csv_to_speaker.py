import numpy as np
import os
import pandas as pd
import argparse

PARAM = 'PARAM'

def load_file_csv(file):
    if os.path.exists(file):
        param_set = pd.read_csv(file)
        return param_set
    else:
        raise ValueError('File %s not found!'%file)

def error_check(param_set):

    if PARAM not in df.columns:
        raise ValueError('[ERROR] %s column is not exist in a dataframe'%PARAM)

    if param_set.shape[1] != 25:
        raise ValueError('[ERROR] Incomplete column provided!, only %s given'%str(param_set.shape[1]))

def create_speaker_from_csv(file, speaker_template_filepath,output_filename):
    '''
    create speaker file using simulated speaker vocaltract
    '''
    param_set = load_file_csv(file)
    error_check(param_set)
    
    syllable_labels = param_set[PARAM].values
    param_set = param_set.drop([PARAM], axis=1)
    syllable_params = param_set.values
    param_names = param_set.columns.values

    with open(speaker_template_filepath[0],'r') as f:
        speaker_head = f.read()

    with open(speaker_template_filepath[1],'r') as f:
        speaker_tail = f.read()

    f = open(output_filename, 'w')
    f.write(speaker_head)
    # Loop through each parameter in the list
    for idx, label in enumerate(syllable_labels):
        f.write('<shape name="%s">\n'%(label))
        for jdx, param in enumerate(syllable_params[idx]):
            f.write('<param name="%s" value="%.2f"/>\n'%(param_names[jdx],param))
        f.write('</shape>\n')
    # Close with tail part of the speaker file
    f.write(speaker_tail)
    f.close()

def main(args):

    if not os.path.exists(args.csv_path):
        raise ValueError('[ERROR] CSV file %s does not exist'%args.csv_path)
    if not os.path.exists(args.head_filepath):
        raise ValueError('[ERROR] Speaker header template %s does not exist'%args.head_filepath)
    if not os.path.exists(args.tail_filepath):
        raise ValueError('[ERROR] Speaker tail template %s does not exist'%args.tail_filepath)

    create_speaker_from_csv(args.csv_path, 
        speaker_template_filepath=[args.speaker_template_head, args.speaker_template_tail],
        output_filename = args.output_filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CSV to Speaker")
    parser.add_argument("csv_path", help="csv file path", type=str)
    parser.add_argument('--head_filepath', dest='speaker_template_head', default='templates/speaker_head.speaker', help='speaker head template file path', type=str)
    parser.add_argument('--tail_filepath', dest='speaker_template_tail', default='templates/speaker_tail.speaker', help='speaker tail template file path', type=str)
    parser.add_argument('--output_filename', dest='output_filename', default='speaker_for_param_test.speaker', help='result speaker filename', type=str)
    args = parser.parse_args()
    main(args)