import numpy as np
import os
import pandas as pd
import argparse

def load_file_csv(file):
    '''
    Load data from csv file.
    '''
    if os.path.exists(file):
        param_set = pd.read_csv(file)
        return param_set
    else:
        raise ValueError('File %s not found!'%file)

def create_speaker_from_csv(file):
    '''
    create speaker file using simulated speaker vocaltract
    '''
    param_set = load_file_csv(file)
    
    params = param_set.values
    # remove digit in parameter name from a columns
    param_name = [''.join(i for i in colname if not i.isdigit()) for colname in param_set.columns.values]
    speaker_tail=open('templates/speaker_tail.txt','r').read() 
    speaker_head=open('templates/speaker_head.speaker','r').read()  

    f = open('test_param.speaker', 'w')
    f.write(speaker_head)
    # Loop through each parameter in the list
    for idx, pair_param in enumerate(params):
        f.write('<shape name="s%s">\n'%(idx))
        for jdx, param in enumerate(pair_param):
            f.write('<param name="%s" value="%.2f"/>\n'%(param_name[jdx],param))
        f.write('</shape>\n')
    # Close with tail part of the speaker file
    f.write(speaker_tail)
    f.close()

def main(args):
    create_speaker_from_csv(args.csv_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("CSV to Speaker")
    parser.add_argument("csv_path", help="csv file path", type=str)
    args = parser.parse_args()
    main(args)