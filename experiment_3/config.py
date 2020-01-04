# Description
EXP_DESCRIPTION = 'New Experiment for Journal using FC, BiLSTM, LSTM, CNN'
# DI_SYLLABLE = False
DI_SYLLABLE = True
#---------------------------------
DATASET_DIR = '../data/d_dataset_2/prep_data'
# DATASET_DIR = '../data/m_dataset_1/prep_data'
#---------------------------------
TEST_SIZE = 0.05
VAL_SIZE = 0.05

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
OPT = 'rmsprop' # for record only

# Training
BATCH_SIZE = 128
EPOCHS = 1000
MODEL_VERBOSE = 0
CHECKPOINT_PEROID = 50
EARLY_STOP_PATIENCE = 10

# Model Type
CNN = False

# --------------------------------
# Evaluation
# --------------------------------
PREP_EVAL_FOLDER = 'prep_data'

EVALSET_DIR = '../data/d_eval'
# MODEL_FILE = '19_d_bilstm_early10.h5'
# EVAL_EXP_NUM = 19

MODEL_FILE = '9_d_bilstm.h5'
EVAL_EXP_NUM = 9

# EVALSET_DIR = '../data/m_eval'
# MODEL_FILE = '10_m_bilstm.h5'
# EVAL_EXP_NUM = 10

#----------------------------------
# Praat config
PRAAT_EXE = r"G:\Praat.exe"

LOG_SHEET = 'log_experiment_journal.csv'