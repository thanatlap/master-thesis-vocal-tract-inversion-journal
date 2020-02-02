# Data generator control
DI_SYLLABLE = True
CONT = False
REPLACE_FOLDER = False
N_SPLIT = 20

# Data generator hyperparameter
DATASIZE =30000
FILTER_THRES = 0.9
SAMPLING_STEP = 0.01
NJOB = 6
MIN_MAX_PERCENT_CHANGE = [0.01, 0.30] # min max
RAMDOM_PARAM_NOISE_PROB = 0.005
SPEAKER_N = [0.0, 0.1, 0.2, 0.25, -0.1, -0.2, -0.25]
# SPEAKER_N = [0.0]
#Data
DATASET_NAME = 'd_dataset_7'
DATA_DESCRIPTION = 'Data for experiment 10 (as of 31 Jan 2020)'
# DATA_DESCRIPTION = 'Testing'
# audio generation
DATASET_DIR = '../data/'+DATASET_NAME


#log
DATA_LOG_FILE = DATASET_DIR+'/data_log.txt'
CLEAN_FILE = False

# required file
VTL_FILE = 'VTL/VocalTractLabApi.dll'
TEMPLATE_DIR = 'templates'
PREDEFINE_PARAM_FILE = TEMPLATE_DIR+'/default_param_set4.csv'
ADULT_SPEAKER_HEADER_FILE = TEMPLATE_DIR+'/adult_speaker_header.txt'
INFANT_SPEAKER_HEADER_FILE = TEMPLATE_DIR+'/infant_speaker_header.txt'
TAIL_SPEAKER = TEMPLATE_DIR+'/speaker_tail.txt'
LABEL_NAME = TEMPLATE_DIR+'/syllable_name.txt'
GES_HEAD = TEMPLATE_DIR+'/ges_head.txt'