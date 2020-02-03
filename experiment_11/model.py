import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import InputLayer, AlphaDropout, Activation, BatchNormalization, Dropout, Flatten, Dense, Bidirectional, LSTM, Conv1D
import config as cf
from functools import partial

N_OUTPUTS = 17

selfNormDense = partial(Dense, 
	activation='selu', 
	kernel_initializer='lecun_normal',
	kernel_regularizer=keras.regularizers.l2(0.01))

regDense = partial(Dense, 
	activation='elu', 
	kernel_initializer='he_normal',
	kernel_regularizer=keras.regularizers.l2(0.01))

bDense = partial(Dense, 
	activation='linear', 
	kernel_initializer='he_normal')

pLSTM = partial(LSTM,
	kernel_initializer='he_normal',
	return_sequences=True)

def rmse(y_true, y_pred):
	return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

def huber_loss(y_true, y_pred):
	error = y_true-y_pred
	is_small_error = tf.abs(error) < 0.31
	square_loss = tf.square(error)
	linear_loss = tf.abs(error)
	return tf.where(is_small_error, square_loss, linear_loss)

def self_norm_fc(input_shape_1,input_shape_2):

	dropout_rate = 0.3
	unit_ff = 1024
	layer_num = 6

	model = tf.keras.Sequential()
	model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
	for i in range(layer_num):
		model.add(selfNormDense(unit_ff))
		model.add(AlphaDropout(rate=dropout_rate))
	model.add(selfNormDense(N_OUTPUTS, activation='linear'))
	model.summary()
	return model

def fc_large_batchnorm(input_shape_1,input_shape_2):

	unit_ff = 1024
	layer_num = 8

	model = tf.keras.Sequential()
	model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
	for i in range(layer_num):
		model.add(bDense(unit_ff))
		model.add(BatchNormalization())
		model.add(Activation('elu'))
	model.add(bDense(N_OUTPUTS, activation='linear'))
	model.summary()
	return model

def inti_bilstm(unit_lstm=128, dropout_rate=0.5, bi_layer_num=5):

	pLSTM = partial(LSTM,
		kernel_initializer='he_normal',
		return_sequences=True)

	def bilstm(input_shape_1,input_shape_2):
		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		# feature extraction layers
		for i in range(bi_layer_num):
			model.add(Bidirectional(pLSTM(unit_lstm)))
			model.add(Dropout(rate=dropout_rate))
		# output layers
		model.add(pLSTM(N_OUTPUTS, activation='linear', return_sequences=False))
		model.summary()
		return model

	return bilstm

def inti_cnn_bilstm(unit_cnn = 64, unit_lstm=64, cnn_filter=3, 
	dropout_rate=0.4, cnn_layer = 3, bi_layer_num=5):

	pLSTM = partial(LSTM,
		kernel_initializer='he_normal',
		return_sequences=True)

	pConv1D = partial(Conv1D,
		padding = 'same',
		activation = 'elu',
		kernel_initializer = 'he_normal')

	def cnn_bilstm(input_shape_1,input_shape_2):

		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		
		# cnn as feature extraction
		for i in range(cnn_layer):
			model.add(pConv1D(filters = unit_cnn, kernel_size= cnn_filter))

		# feature extraction layers
		for j in range(bi_layer_num):
			model.add(Bidirectional(pLSTM(unit_lstm)))
			model.add(Dropout(rate=dropout_rate))

		# output layers
		model.add(pLSTM(N_OUTPUTS, activation='linear', return_sequences=False))
		model.summary()
		return model

	return cnn_bilstm

def inti_cnn_fc(unit_cnn = 64, unit_dense=1024, cnn_filter=3, dropout_rate=0.4, 
	cnn_layer = 5, dense_layer_num=1):

	pConv1D = partial(Conv1D,
		padding = 'same',
		activation = 'elu',
		kernel_initializer = 'he_normal')

	pDense = partial(Dense, 
		activation='elu', 
		kernel_initializer='he_normal')

	def cnn_fc(input_shape_1,input_shape_2):

		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		
		# cnn as feature extraction
		for i in range(cnn_layer):
			model.add(pConv1D(filters = unit_cnn, kernel_size= cnn_filter))
			model.add(Dropout(rate=dropout_rate))

		# FC 
		model.add(Flatten())
		for i in range(dense_layer_num):
			model.add(pDense(unit_dense))
			model.add(Dropout(rate=dropout_rate))

		# output layers
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model

	return cnn_fc