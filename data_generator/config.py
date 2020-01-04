# data type
DI_SYLLABLE = False
CONT = True

#Data generator control
DATASIZE =2000 # this size will be multplied by n speaker
N_SPLIT = 5

#Config
SAMPLING_STEP = 0.01
# SOUND_SAMPLING_RATE = 16000 # for experiment 3
SOUND_SAMPLING_RATE = 8000 # for experiment 4 & 5
NJOB = 6

#Data
DATASET_NAME = 'm_dataset_3_u_L'
DATA_DESCRIPTION = 'Adding Data\nData for new experiment (as of 03 Jan 2020)'
# DATA_DESCRIPTION = 'Testing'
# audio generation
DATASET_DIR = '../data/'+DATASET_NAME
VTL_FILE = 'VTL/VocalTractLabApi.dll'
TEMPLATE_DIR = 'templates'
PREDEFINE_PARAM_FILE = TEMPLATE_DIR+'/default_param_small_u_ver.csv'

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