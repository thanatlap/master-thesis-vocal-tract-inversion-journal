import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import AlphaDropout, Activation, BatchNormalization, Dropout, Flatten, Dense, Bidirectional, LSTM
import config as cf
from functools import partial

N_OUTPUTS = 17

selfNormDense = partial(Dense, 
	activation='selu', 
	kernel_initializer='lecun_normal',
	kernel_regularizer=keras.regularizers.l2(0.01))

pDense = partial(Dense, 
	activation='elu', 
	kernel_initializer='he_normal',
	kernel_regularizer=keras.regularizers.l2(0.01))

bDense = partial(Dense, 
	activation='linear', 
	kernel_initializer='he_normal',
	kernel_regularizer=keras.regularizers.l2(0.01))

pLSTM = partial(LSTM,
	kernel_initializer='he_normal',
	return_sequences=True)

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def self_norm_fc(input_shape_1,input_shape_2):

	dropout_rate = 0.3
	unit_ff = 1024

	model = tf.keras.Sequential()
	model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
	model.add(selfNormDense(unit_ff))
	model.add(AlphaDropout(rate=dropout_rate))
	model.add(selfNormDense(unit_ff))
	model.add(AlphaDropout(rate=dropout_rate))
	model.add(selfNormDense(unit_ff))
	model.add(AlphaDropout(rate=dropout_rate))
	model.add(selfNormDense(unit_ff))
	model.add(AlphaDropout(rate=dropout_rate))
	model.add(selfNormDense(unit_ff))
	model.add(AlphaDropout(rate=dropout_rate))
	model.add(selfNormDense(unit_ff))
	model.add(AlphaDropout(rate=dropout_rate))
	model.add(selfNormDense(N_OUTPUTS, activation='linear'))
	model.summary()
	return model

def fc(input_shape_1,input_shape_2):

	dropout_rate = 0.3
	unit_ff = 1024

	model = tf.keras.Sequential()
	model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(N_OUTPUTS, activation='linear'))
	model.summary()
	return model

def fc_large(input_shape_1,input_shape_2):

	dropout_rate = 0.4
	unit_ff = 1024

	model = tf.keras.Sequential()
	model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(unit_ff))
	model.add(pDense(N_OUTPUTS, activation='linear'))
	model.summary()
	return model

def fc_large_batchnorm(input_shape_1,input_shape_2):

	unit_ff = 512
	layer_num = 15

	model = tf.keras.Sequential()
	model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
	for i in range(layer_num):
		model.add(bDense(unit_ff))
		model.add(BatchNormalization())
		model.add(Activation('elu'))
	model.add(pDense(N_OUTPUTS, activation='linear'))
	model.summary()
	return model

def bilstm_1(input_shape_1,input_shape_2):

	unit_lstm = 64
	unit_ff = 1024
	dropout_rate = 0.5

	model = tf.keras.Sequential()
	model.add(Bidirectional(pLSTM(unit_lstm, input_shape=(input_shape_1,input_shape_2))))
	model.add(Dropout(rate=dropout_rate))
	model.add(Bidirectional(pLSTM(unit_lstm)))
	model.add(Dropout(rate=dropout_rate))
	model.add(Bidirectional(pLSTM(unit_lstm)))
	model.add(Dropout(rate=dropout_rate))
	model.add(Bidirectional(pLSTM(unit_lstm, return_sequences=False)))
	model.add(Dropout(rate=dropout_rate))
	model.add(pDense(unit_ff))
	model.add(Dropout(rate=dropout_rate))
	model.add(Dense(N_OUTPUTS, activation='linear'))
	return model

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
	r2 = R2(y_true, y_pred)
	return (1-r2)*mse + mse

def cus_loss2(y_true, y_pred):
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	r2 = R2(y_true, y_pred)
	return (1-r2)*K.sqrt(mse) + mse

def cus_loss3(y_true, y_pred):
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	r2 = R2(y_true, y_pred)
	return (1-r2)*mse + K.sqrt(mse)

def cus_loss4(y_true, y_pred):
	rmse = K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))
	r2 = R2(y_true, y_pred)
	return (1-r2)*rmse + rmse
