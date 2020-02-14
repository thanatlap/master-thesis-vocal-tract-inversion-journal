import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Reshape, GRU, InputLayer, AlphaDropout, Activation, BatchNormalization, Dropout, Flatten, Dense, Bidirectional, LSTM, Conv1D
import config as cf
from functools import partial

N_OUTPUTS = 17

pDense = partial(Dense,
		kernel_initializer='he_normal',
		activation='elu')

pLSTM = partial(LSTM,
		kernel_initializer='he_normal',
		return_sequences=True)

pConv1D = partial(Conv1D,
		padding = 'same',
		activation = 'linear',
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

def init_FCNN(unit_ff = 1024, layer_num = 4, drop_rate=0.4):

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

def init_bilstm(unit=128, bi_layer_num=5, drop_rate=0.4):

	def bilstm(input_shape_1,input_shape_2):
		model = tf.keras.Sequential(InputLayer(input_shape=(input_shape_1,input_shape_2)))
		# feature extraction layers
		for i in range(bi_layer_num-1):
			model.add(Bidirectional(pLSTM(unit)))
			if drop_rate: model.add(Dropout(rate=drop_rate))
		# output layers
		model.add(Bidirectional(pLSTM(unit, return_sequences=False)))
		if drop_rate: model.add(Dropout(rate=drop_rate))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		model.summary()
		return model
	return bilstm


def residual_block(input_x, units=64):
		
	x = pConv1D(units, kernel_size=3)(input_x)
	x = BatchNormalization()(x)
	x = Activation('elu')(x)
	x = pConv1D(units, kernel_size=3)(x)
	x = BatchNormalization()(x)
	x = keras.layers.Add()([x, input_x])
	output = Activation('elu')(x)
	return output


def init_resnet(large=False):

	if large:
		res_layers = [64, 64, 64, 64, 64, 64, 64, 64, 64, 64]
	else:
		res_layers = [64, 64, 64]

	def resnet(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x = pConv1D(64, kernel_size=7)(input_x)
		x = Activation('elu')(x)
		x = BatchNormalization()(x)
		x = Activation('elu')(x)
		for idx, unit in enumerate(res_layers):
			x = residual_block(x, units=unit)
		x = layers.GlobalAveragePooling1D()(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		model.summary()

		return model
	return resnet

def init_cnn_bilstm(feature_layer=3, bilstm_layer=3):

	def cnn_bilstm(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x = pConv1D(64, kernel_size=7)(input_x)
		x = BatchNormalization()(x)
		x = Activation('elu')(x)
		for i in range(feature_layer):
			x = residual_block(x)
		for i in range(bilstm_layer-1):
			x = Bidirectional(pLSTM(128))(x)
			x = layers.SpatialDropout1D(rate=0.4)(x)
		x = Bidirectional(pLSTM(128, return_sequences=False))(x)
		x = layers.Dropout(rate=0.4)(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		model.summary()
		return model
	return cnn_bilstm


def init_senet(feature_layer=3, bilstm = False, dense=False):

	reduction_ratio = 4

	def se_block(input_x):
		x = layers.GlobalAveragePooling1D()(input_x)
		channel_shape = getattr(x, '_shape_val')[-1]
		x = Reshape((1, channel_shape))(x)
		x = Dense(channel_shape // reduction_ratio, activation='elu', kernel_initializer='he_normal')(x)
		outputs = Dense(channel_shape, activation='sigmoid', kernel_initializer='he_normal')(x)
		return outputs

	def residual_block(input_x):
		
		x = pConv1D(64, kernel_size=3)(input_x)
		x = BatchNormalization()(x)
		x = Activation('elu')(x)
		x = pConv1D(64, kernel_size=3)(x)
		x = BatchNormalization()(x)
		outputs = Activation('elu')(x)
		return outputs

	def se_res_block(input_x):

		res_x = residual_block(input_x)
		se_x = se_block(res_x)
		x = layers.Multiply()([res_x, se_x])
		outputs = layers.Add()([x, input_x])
		return outputs

	def res(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x = pConv1D(64, kernel_size=7)(input_x)
		x = BatchNormalization()(x)
		x = Activation('elu')(x)
		for i in range(feature_layer):
			x = se_res_block(x)

		if bilstm:
			x = Bidirectional(pLSTM(128))(x)
			x = layers.SpatialDropout1D(rate=0.4)(x)
			x = Bidirectional(pLSTM(128))(x)
			x = layers.SpatialDropout1D(rate=0.4)(x)
			x = Bidirectional(pLSTM(128, return_sequences=False))(x)
			x = layers.Dropout(rate=0.4)(x)
		else:
			x = layers.GlobalAveragePooling1D()(x)
		if dense: 
			x = pDense(1024, activation='elu')(x)
			x = layers.Dropout(rate=0.4)(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		model.summary()
		return model
	return res


def init_LTRCNN(drop_rate=None):

	def LTRCNN(input_shape_1,input_shape_2):
		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x1 = pConv1D(128, kernel_size=3, activation='relu')(input_x)
		if drop_rate: x1 = Dropout(rate=drop_rate)(x1)
		x2 = pConv1D(128, kernel_size=3, activation='relu')(x1)
		if drop_rate: x2 = Dropout(rate=drop_rate)(x2)
		x = layers.Concatenate()([x1, x2])
		x = pLSTM(1024, return_sequences=False)(x)
		if drop_rate: x = Dropout(rate=drop_rate)(x)
		x = pDense(1024, activation='relu')(x)
		if drop_rate: x = Dropout(rate=drop_rate)(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		model.summary()
		return model
	return LTRCNN