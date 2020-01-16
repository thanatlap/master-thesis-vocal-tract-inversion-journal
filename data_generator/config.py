# data type
DI_SYLLABLE = True
CONT = False

#Data generator control
DATASIZE =100
N_SPLIT = 1
FILTER_THRES = 0.85
#Config
SAMPLING_STEP = 0.01
NJOB = 6

#Data
DATASET_NAME = 'd_dataset_0'
# DATA_DESCRIPTION = 'Data for experiment 9 (as of 15 Jan 2020)'
DATA_DESCRIPTION = 'Testing'
# audio generation
DATASET_DIR = '../data/'+DATASET_NAME
VTL_FILE = 'VTL/VocalTractLabApi.dll'
TEMPLATE_DIR = 'templates'
PREDEFINE_PARAM_FILE = TEMPLATE_DIR+'/default_param_set2.csv'

# speaker simulation
ADULT_SPEAKER_FILE = TEMPLATE_DIR+'/adult_speaker.txt'
INFANT_SPEAKER_FILE = TEMPLATE_DIR+'/infant_speaker.txt'
TAIL_SPEAKER = TEMPLATE_DIR+'/speaker_tail.txt'
SPKEAKER_SIM_DIR = DATASET_DIR+'/speaker_sim'
LABEL_NAME = TEMPLATE_DIR+'/label_name_small.npy'
# the first n_speaker must be 0.0
SPEAKER_N = [0.0, 0.1, 0.2, 0.3, -0.1, -0.2, -0.3]
# SPEAKER_N = [0.0]

# ges creation
GES_HEAD = TEMPLATE_DIR+'/ges_head.txt'

#log
DATA_LOG_FILE = DATASET_DIR+'/data_log.txt'
CLEAN_FILE = False