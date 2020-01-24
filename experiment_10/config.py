# Description
# EXP_DESCRIPTION = 'MONOSYLLABIC WITH FEWER TIMEFRAME'
EXP_DESCRIPTION = 'TEST GENERALIZATION ABILITY'
# DI_SYLLABLE = False
DI_SYLLABLE = True
#---------------------------------
# DATASET_DIR = '../data/d_dataset_4/prep_exp9'
DATASET_DIR = '../data/d_dataset_4/prep_exp10_l4'

#---------------------------------

LOAD_FROM_SAVE = None
LOAD_FROM_CHECKPOINT = None

#---------------------------------
#Experiment setting
#Model compile setting
LEARNING_RATE = 0.005  
BETA1 = 0.9
BETA2 = 0.999
EPS = None
SDECAY = 0.0
AMSGRAD = True
LOSS_FN = 'mse'
OPT = 'RMSProp' # for record only

# Training
BATCH_SIZE = 128
EPOCHS = 200
MODEL_VERBOSE = 2
CHECKPOINT_PEROID = 50
EARLY_STOP_PATIENCE = None # should increase earlystop since it seen can be improve

# Tensorboard
TENSORBOARD = True

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