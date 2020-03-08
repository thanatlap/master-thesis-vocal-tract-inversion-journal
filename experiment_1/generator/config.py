# Data generator control
DI_SYLLABLE = True
CONT = False
REPLACE_FOLDER = False
N_SPLIT = 1
CLEAN_FILE = False
CLEAN_SOUND = False
NJOB = 2
DATASIZE = 10

# Data generator hyperparameter
FILTER_THRES = 0.9
SAMPLING_STEP = 0.01
MIN_MAX_PERCENT_CHANGE = [0.01, 0.30] # min max
RAMDOM_PARAM_NOISE_PROB = 0.00
SPEAKER_N = [0.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3]
AUDIO_SAMPLE_RATE = 16000
GES_MIN_MAX_DURATION_DI = [1.1, 1.5] # min max
GES_MIN_MAX_DURATION_MONO = [0.4, 0.7] # min max
GES_VARY_DURATION_DI = [0.45, 0.55] # %min %max
GES_TIME_CONST = [0.015, 0.02] # min max
GES_F0_INIT_MIN_MAX = [80, 81] # min max
GES_F0_NEXT_MIN_MAX = [81, 83] # min max

#Data
DATASET_NAME = 'd_dataset_0'
DATA_DESCRIPTION = 'Data with 16000 sample rate'
DATASET_DIR = '../../data/'+DATASET_NAME

# required file
TEMPLATE_DIR = 'assets'
VTL_FILE = TEMPLATE_DIR+'/VTL/VocalTractLabApi.dll'
PREDEFINE_PARAM_FILE = TEMPLATE_DIR+'/default_param_draft1.csv'
ADULT_SPEAKER_HEADER_FILE = TEMPLATE_DIR+'/adult_speaker_header.txt'
INFANT_SPEAKER_HEADER_FILE = TEMPLATE_DIR+'/infant_speaker_header.txt'
TAIL_SPEAKER = TEMPLATE_DIR+'/speaker_tail.txt'
GES_HEAD = TEMPLATE_DIR+'/ges_head.txt'
PREDEFINED_PARAM = TEMPLATE_DIR+'/predefined_param.npz'
SIM_PATH = DATASET_DIR + '/simulated_speakers'


