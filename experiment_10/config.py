# Description
# EXP_DESCRIPTION = 'MONOSYLLABIC WITH FEWER TIMEFRAME'
EXP_DESCRIPTION = 'TEST GENERALIZATION ABILITY'
# DI_SYLLABLE = False
DI_SYLLABLE = True
#---------------------------------
DATASET_DIR = '../data/d_dataset_4L/prep_exp10_XL_2'

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
AMSGRAD = True
LOSS_FN = 'mse'
OPT = 'AMSGRAD' # for record only
OPT_NUM = 1

# Training
BATCH_SIZE = 64
EPOCHS = 1000
MODEL_VERBOSE = 2
CHECKPOINT_PEROID = 50
EARLY_STOP_PATIENCE = 15 # should increase earlystop since it seen can be improve

# Tensorboard
TENSORBOARD = False

# --------------------------------
# Evaluation
# --------------------------------
PREP_EVAL_FOLDER = 'prep_exp9_2'

EVALSET_DIR = '../data/d_eval'
MODEL_FILE = '257_bilstm_2_20200120_1026.h5'
EVAL_EXP_NUM = 257

LABEL_MODE = 2 #1:standardized 2:minmax

#----------------------------------
# Praat config
PRAAT_EXE = r"G:\Praat.exe"

# LOG_SHEET = 'log_experiment_journal.csv'
LOG_SHEET = 'exp10_attemp2.csv'