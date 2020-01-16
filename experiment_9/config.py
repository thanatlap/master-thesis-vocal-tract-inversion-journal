# Description
# EXP_DESCRIPTION = 'MONOSYLLABIC WITH FEWER TIMEFRAME'
EXP_DESCRIPTION = 'DISYLLABIC WITH NEW METHOD (EXP7)'
# DI_SYLLABLE = False
DI_SYLLABLE = True
#---------------------------------
# DATASET_DIR = '../data/d_dataset_2/prep_data'
# DATASET_DIR = '../data/m_dataset_2_sm/prep_data'
# DATASET_DIR = '../data/m_dataset_3_u_L/prep_data_exp7'
# DATASET_DIR = '../data/d_dataset_3_u/prep_data_exp7'
DATASET_DIR = '../data/d_dataset_4/prep_exp9'


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
OPT = 'AMSGrad' # for record only

# Training
BATCH_SIZE = 128
EPOCHS = 1000
MODEL_VERBOSE = 0
CHECKPOINT_PEROID = 50
EARLY_STOP_PATIENCE = 5 # should increase earlystop since it seen can be improve

# Model Type
CNN = False

# --------------------------------
# Evaluation
# --------------------------------
PREP_EVAL_FOLDER = 'prep_data_exp7'

EVALSET_DIR = '../data/d_eval'
MODEL_FILE = '157_nn_bilstm_12_20200108_0020.h5'
EVAL_EXP_NUM = 157

# MODEL_FILE = '9_d_bilstm.h5'
# EVAL_EXP_NUM = 9

# EVALSET_DIR = '../data/m_eval'
# MODEL_FILE = '124_nn_fbc_5_20200106_1456.h5'
# EVAL_EXP_NUM = 124

LABEL_MODE = 1 #1:standardized 2:minmax

#----------------------------------
# Praat config
PRAAT_EXE = r"G:\Praat.exe"

LOG_SHEET = 'log_experiment_journal.csv'