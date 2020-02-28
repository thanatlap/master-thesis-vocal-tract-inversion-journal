# Description
EXP_DESCRIPTION = 'EXP'
# DI_SYLLABLE = False
DI_SYLLABLE = True
#---------------------------------
DATASET_DIR = '../data/d_dataset_t3/prep_data_v2'

#---------------------------------

LOAD_FROM_SAVE =None
LOAD_FROM_CHECKPOINT = None

#---------------------------------
#Experiment setting
#Model compile setting
LEARNING_RATE = 0.001  
BETA1 = 0.9
BETA2 = 0.999
EPS = None
SDECAY = 0.0
AMSGRAD = True
LOSS_FN = 'mse'
OPT = 'AMSgrad' # for record only
OPT_NUM = 1

# Training
BATCH_SIZE = 128
EPOCHS = 50
MODEL_VERBOSE = 2
CHECKPOINT_PEROID = 50
EARLY_STOP_PATIENCE = None # should increase earlystop since it seen can be improve

# Tensorboard
TENSORBOARD = False

# --------------------------------
# Evaluation
# --------------------------------
PREP_EVAL_FOLDER = 'prep_exp11'

EVALSET_DIR = '../data/d_eval'
MODEL_FILE = '8_cnn_bilstm.h5'

LABEL_MODE = 3 #1:standardized 2:minmax

#----------------------------------
# Praat config
PRAAT_EXE = r"G:\Praat.exe"
LOG_SHEET = 'thesis_log.csv'