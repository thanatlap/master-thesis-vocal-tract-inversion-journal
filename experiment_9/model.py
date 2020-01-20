import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, BatchNormalization 
import config as cf

N_OUTPUTS = 17

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def R2(y_true, y_pred):
	SS_res =  K.sum(K.square( y_true-y_pred ))
	SS_tot = K.sum(K.square( y_true - K.mean(y_true) ) )
	return  ( 1 - SS_res/(SS_tot + K.epsilon()) )
	
def mse(y_true, y_pred):
	return K.mean(K.square(y_pred - y_true), axis=-1)

def AdjustR2(y_true, y_pred):
	r2 = R2(y_true, y_pred)
	N = cf.BATCH_SIZE
	p = K.int_shape(y_pred)[1]
	return 1 - ((1-r2)*(N-1)/(N-p-1))

def cus_loss1(y_true, y_pred):
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	adjustR2 = AdjustR2(y_true, y_pred)
	return (1-adjustR2)*mse + mse

def cus_loss2(y_true, y_pred):
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	adjustR2 = AdjustR2(y_true, y_pred)
	return (1-adjustR2)*K.sqrt(mse) + mse

def cus_loss3(y_true, y_pred):
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	adjustR2 = AdjustR2(y_true, y_pred)
	return (1-adjustR2)*mse + K.sqrt(mse)

def cus_loss4(y_true, y_pred):
	rmse = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
	adjustR2 = AdjustR2(y_true, y_pred)
	return (1-adjustR2)*rmse + rmse


def nn_fc(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	
	model.add(layers.Flatten(input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(N_OUTPUTS, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def bilstm_1(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(64)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(256, activation='relu' ))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(N_OUTPUTS, activation='linear'))
	return model

def bilstm_2(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(32)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(256, activation='relu' ))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(N_OUTPUTS, activation='linear'))
	return model

def bilstm_3(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(32)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(256, activation='elu' ))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(N_OUTPUTS, activation='linear'))
	return model

def lstm_1(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.LSTM(32, return_sequences=True, input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.LSTM(32, return_sequences=True))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.LSTM(32, return_sequences=True))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.LSTM(32))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(256, activation='relu' ))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(N_OUTPUTS, activation='linear'))
	return model

def bilstm_2(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(32, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(32)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(256, activation='elu' ))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(N_OUTPUTS, activation='linear'))
	return model