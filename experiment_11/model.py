import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import GRU, InputLayer, AlphaDropout, Activation, BatchNormalization, Dropout, Flatten, Dense, Bidirectional, LSTM, Conv1D
import config as cf
from functools import partial

N_OUTPUTS = 17

pDense = partial(Dense,
		kernel_initializer='he_normal',
		activation='elu')

pLSTM = partial(LSTM,
		kernel_initializer='he_normal',
		return_sequences=True)

pGRU = partial(GRU,
		kernel_initializer='he_normal',
		return_sequences=True)

pConv1D = partial(Conv1D,
		padding = 'same',
		activation = 'elu',
		kernel_initializer = 'he_normal')

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def init_baseline():

	def baseline(input_shape_1,input_shape_2):
		model = tf.keras.Sequential()
		model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model
	return baseline

def init_FCNN(unit_ff = 1024, layer_num = 5, drop_rate=None):

	def FCNN(input_shape_1,input_shape_2):
		model = tf.keras.Sequential()
		model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
		for i in range(layer_num):
			model.add(pDense(unit_ff))
			if drop_rate: model.add(Dropout(rate=drop_rate))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model
	return FCNN

def inti_lstm(unit=128, layer_num=5, drop_rate=None):

	def lstm(input_shape_1,input_shape_2):
		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		# feature extraction layers
		for i in range(layer_num-1):
			model.add(pLSTM(unit))
			if drop_rate: model.add(Dropout(rate=drop_rate))
		# output layers
		model.add(pLSTM(unit, return_sequences=False))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model
	return lstm

def inti_gru(unit=128, layer_num=5, drop_rate=None):

	def gru(input_shape_1,input_shape_2):
		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		# feature extraction layers
		for i in range(layer_num-1):
			model.add(pGRU(unit))
			if drop_rate: model.add(Dropout(rate=drop_rate))
		# output layers
		model.add(pGRU(unit, return_sequences=False))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model
	return gru

def inti_bilstm(unit=128, bi_layer_num=5, drop_rate=None):

	def bilstm(input_shape_1,input_shape_2):
		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		# feature extraction layers
		for i in range(bi_layer_num-1):
			model.add(Bidirectional(pLSTM(unit)))
			if drop_rate: model.add(Dropout(rate=drop_rate))
		# output layers
		model.add(Bidirectional(pLSTM(unit, return_sequences=False)))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model
	return bilstm

def inti_bigru(unit=128, bi_layer_num=5, drop_rate=None):

	def bigru(input_shape_1,input_shape_2):
		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		# feature extraction layers
		for i in range(bi_layer_num-1):
			model.add(Bidirectional(pGRU(unit)))
			if drop_rate: model.add(Dropout(rate=drop_rate))
		# output layers
		model.add(Bidirectional(pGRU(unit, return_sequences=False)))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model
	return bigru

def inti_cnn_bilstm(unit_cnn = 64, cnn_filter=3, cnn_layer = 3, 
	unit_lstm=128, bi_layer_num=4, drop_rate=None):

	def cnn_bilstm(input_shape_1,input_shape_2):

		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		
		# cnn as feature extraction
		for i in range(cnn_layer):
			model.add(pConv1D(filters = unit_cnn, kernel_size= cnn_filter))

		# feature extraction layers
		for j in range(bi_layer_num-1):
			model.add(Bidirectional(pLSTM(unit_lstm)))
			if drop_rate: model.add(Dropout(rate=drop_rate))

		# output layers
		model.add(Bidirectional(pLSTM(unit, return_sequences=False)))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model

	return cnn_bilstm

def inti_cnn_bigru(unit_cnn = 64, cnn_filter=3, cnn_layer = 3, 
	unit_gru=128, bi_layer_num=4, drop_rate=None):

	def cnn_bigru(input_shape_1,input_shape_2):

		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		
		# cnn as feature extraction
		for i in range(cnn_layer):
			model.add(pConv1D(filters = unit_cnn, kernel_size= cnn_filter))

		# feature extraction layers
		for j in range(bi_layer_num-1):
			model.add(Bidirectional(pGRU(unit_gru)))
			if drop_rate: model.add(Dropout(rate=drop_rate))

		# output layers
		model.add(Bidirectional(pGRU(unit, return_sequences=False)))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model

	return cnn_bigru

