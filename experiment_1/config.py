# Description
EXP_DESCRIPTION = 'New Experiment for Journal using BiLSTMRNN'
MODEL_DETAIL = 'with early stop, decrease earlystop to 15'
DI_SYLLABLE = True
#---------------------------------
DATASET_DIR = '../data/d_dataset_1/prep_data_rnn'

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

# Training
BATCH_SIZE = 128
EPOCHS = 2000
MODEL_VERBOSE = 0
CHECKPOINT_PEROID = 50
EARLY_STOP_PATIENCE = 15

# --------------------------------
# Evaluation
# --------------------------------
EVALSET_DIR = '../data/d_eval'
PREP_EVAL_FOLDER = 'prep_data_rnn'
MODEL_FILE = 'model_20191107_19.hdf5'

#----------------------------------
# Praat config
PRAAT_EXE = r"G:\Praat.exe"