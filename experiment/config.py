# DI_SYLLABLE = False
DI_SYLLABLE = True
#---------------------------------
DATASET_DIR = '../data/d_dataset_p2/prep_data_13'
#---------------------------------

LOAD_FROM_SAVE = None
LOAD_FROM_CHECKPOINT = None

#---------------------------------
#Experiment setting
#Model compile setting
LEARNING_RATE = 0.001  
BETA1 = 0.9
BETA2 = 0.999
EPS = None
SDECAY = 0.0
AMSGRAD = False
LOSS_FN = 'mse'
OPT = 'Adam' # for record only
OPT_NUM = 1

# Training
BATCH_SIZE = 64
EPOCHS = 200
MODEL_VERBOSE = 2
CHECKPOINT_PEROID = None
EARLY_STOP_PATIENCE = 7 
TENSORBOARD = False
SAVE_BEST_WEIGHT = False

# --------------------------------
# Evaluation
# --------------------------------
PREP_EVAL_FOLDER = 'prep_data_13'

EVALSET_DIR = '../data/d_eval'
MODEL_FILE = '39_senet.h5'

LABEL_MODE = 3 #1:standardized 2:minmax

#----------------------------------
# Praat config
PRAAT_EXE = r"G:\Praat.exe"
LOG_SHEET = 'experiment/thesis_log.csv'