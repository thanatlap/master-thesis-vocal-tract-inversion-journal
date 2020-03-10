import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
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

def R2(y_true, y_pred):
	SS_res =  K.sum(K.square(y_true - y_pred), axis = 0) 
	SS_tot = K.sum(K.square(y_true - K.mean(y_true, axis = 0)), axis = 0) 
	return K.mean(1 - (SS_res/SS_tot), axis=0)

def init_baseline():
	def baseline(input_shape_1,input_shape_2):
		model = tf.keras.Sequential()
		model.add(Flatten(input_shape=(input_shape_1,input_shape_2)))
		model.add(pDense(N_OUTPUTS, activation='linear'))
		# model.summary()
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
		# model.summary()
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
		# model.summary()
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


def init_resnet(res_layers=2):

	def resnet(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x = pConv1D(64, kernel_size=7)(input_x)
		x = Activation('elu')(x)
		x = BatchNormalization()(x)
		x = Activation('elu')(x)
		for i in res_layers:
			x = residual_block(x)
		x = layers.GlobalAveragePooling1D()(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		# model.summary()

		return model
	return resnet

def init_res_bilstm(feature_layer=3, bilstm_layer=2):

	def res_bilstm(input_shape_1,input_shape_2):

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
		# model.summary()
		return model
	return res_bilstm


def init_senet(feature_layer=1, cnn_unit=64, cnn_kernel=5, res_unit=128,
	bilstm = 1, bilstm_unit=256, 
	dense=None, 
	dropout_rate=0.3,
	reduction_ratio = 2):


	def cnn_block(input_x, cnn_unit, kernel_size):
		x = pConv1D(cnn_unit, kernel_size=kernel_size)(input_x)
		x = BatchNormalization()(x)
		x = Activation('relu')(x)
		return x

	def se_block(input_x):
		x = layers.GlobalAveragePooling1D()(input_x)
		channel_shape = getattr(x, '_shape_val')[-1]
		x = Reshape((1, channel_shape))(x)
		x = Dense(channel_shape // reduction_ratio, activation='relu', kernel_initializer='he_uniform')(x)
		x = Dense(channel_shape, activation='tanh', kernel_initializer='he_uniform')(x)
		return x

	def residual_block(input_x):
		x = cnn_block(input_x, res_unit,kernel_size=3)
		x = pConv1D(res_unit, kernel_size=3)(x)
		x = BatchNormalization()(x)
		return x

	def se_res_block(input_x):
		res_x = residual_block(input_x)
		se_x = se_block(res_x)
		x = layers.Multiply()([res_x, se_x])
		x = Activation('relu')(x)
		# input_x = cnn_block(input_x, cnn_unit=cnn_unit, kernel_size=1)
		x = layers.Concatenate()([x, input_x])
		outputs = cnn_block(x, cnn_unit=res_unit, kernel_size=1)
		return outputs

	def senet_nn(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x1 = cnn_block(input_x, cnn_unit=cnn_unit, kernel_size=13)
		x2 = cnn_block(x1, cnn_unit=cnn_unit, kernel_size=11)
		x12 = layers.Concatenate()([x1, x2])
		x3 = cnn_block(x12, cnn_unit=cnn_unit, kernel_size=9)
		x123 = layers.Concatenate()([x1, x2, x3])
		x4 = cnn_block(x123, cnn_unit=cnn_unit, kernel_size=7)
		x1234 = layers.Concatenate()([x1, x2, x3, x4])
		x5 = cnn_block(x1234, cnn_unit=cnn_unit, kernel_size=5)
		x12345 = layers.Concatenate()([x1, x2, x3, x4])
		x = cnn_block(x12345, cnn_unit=res_unit, kernel_size=1)
		x = layers.SpatialDropout1D(rate=dropout_rate)(x)
		for i in range(feature_layer):
			x = se_res_block(x)
			x = layers.SpatialDropout1D(rate=dropout_rate)(x)
		x = cnn_block(x, cnn_unit=cnn_unit, kernel_size=1)
		x = layers.SpatialDropout1D(rate=dropout_rate)(x)
		if bilstm:
			for i in range(bilstm-1):
				x = Bidirectional(pLSTM(bilstm_unit))(x)
				x = layers.SpatialDropout1D(rate=dropout_rate)(x)
			x = Bidirectional(pLSTM(bilstm_unit, return_sequences=False))(x)
			x = layers.Dropout(rate=dropout_rate)(x)
		else:
			x = layers.GlobalAveragePooling1D()(x)
		if dense: 
			x = pDense(dense, activation='elu')(x)
			x = layers.Dropout(rate=dropout_rate)(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		model.summary()
		return model

	return senet_nn


def init_senet_skip(feature_layer=1, cnn_unit=64, cnn_kernel=5, 
	bilstm = 2, bilstm_unit=128, reduction_ratio = 2, dense=None, drop_rate=0.4, activation='elu'):

	

	def se_block(input_x):
		x = layers.GlobalAveragePooling1D()(input_x)
		channel_shape = getattr(x, '_shape_val')[-1]
		x = Reshape((1, channel_shape))(x)
		x = Dense(channel_shape // reduction_ratio, activation=activation, kernel_initializer='he_normal')(x)
		outputs = Dense(channel_shape, activation='sigmoid', kernel_initializer='he_normal')(x)
		return outputs

	def residual_block(input_x):
		
		x = pConv1D(cnn_unit, kernel_size=3)(input_x)
		x = BatchNormalization()(x)
		x = Activation(activation)(x)
		x = pConv1D(cnn_unit, kernel_size=3)(x)
		x = BatchNormalization()(x)
		outputs = Activation(activation)(x)
		return outputs

	def se_res_block(input_x):

		res_x = residual_block(input_x)
		se_x = se_block(res_x)
		x = layers.Multiply()([res_x, se_x])
		add = layers.Add()([x, input_x])
		x = layers.Concatenate()([add, x, input_x])
		outputs = pConv1D(cnn_unit, kernel_size=1)(x)
		return outputs

	def res(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x = pConv1D(cnn_unit, kernel_size=cnn_kernel)(input_x)
		x = BatchNormalization()(x)
		x = Activation(activation)(x)
		for i in range(feature_layer):
			x = se_res_block(x)

		if bilstm:
			for i in range(bilstm-1):
				x = Bidirectional(pLSTM(bilstm_unit))(x)
				if drop_rate: x = layers.SpatialDropout1D(rate=drop_rate)(x)
			x = Bidirectional(pLSTM(bilstm_unit, return_sequences=False))(x)
			if drop_rate: x = layers.Dropout(rate=drop_rate)(x)
		else:
			x = layers.GlobalAveragePooling1D()(x)
		if dense: 
			for i in range(dense):
				x = pDense(1024, activation=activation)(x)
			if drop_rate: x = layers.Dropout(rate=drop_rate)(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		# model.summary()
		return model
	return res






def init_LTRCNN(drop_rate=None):

	def LTRCNN(input_shape_1,input_shape_2):
		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x1 = pConv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(input_x)
		if drop_rate: x1 = Dropout(rate=drop_rate)(x1)
		x2 = pConv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x1)
		if drop_rate: x2 = Dropout(rate=drop_rate)(x2)
		x = layers.Concatenate()([x1, x2])
		x = pLSTM(1024, return_sequences=False)(x)
		if drop_rate: x = Dropout(rate=drop_rate)(x)
		x = pDense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
		if drop_rate: x = Dropout(rate=drop_rate)(x)
		outputs = pDense(N_OUTPUTS, activation='linear', kernel_regularizer=regularizers.l2(0.0005))(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		# model.summary()
		return model
	return LTRCNN


def init_densenet(feature_layer=1, cnn_unit=64, cnn_concat_unit = 128,
	bilstm = 1, bilstm_unit=256, 
	dense=None, 
	dropout_rate=0.3,
	reduction_ratio = 2):


	def cnn_block(input_x, cnn_unit, kernel_size):
		x = pConv1D(cnn_unit, kernel_size=kernel_size)(input_x)
		x = BatchNormalization()(x)
		x = Activation('elu')(x)
		return x

	def dense_nn(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x1 = cnn_block(input_x, cnn_unit=cnn_unit, kernel_size=13)
		x2 = cnn_block(x1, cnn_unit=cnn_unit, kernel_size=11)
		x12 = layers.Concatenate()([x1, x2])
		x3 = cnn_block(x12, cnn_unit=cnn_unit, kernel_size=9)
		x123 = layers.Concatenate()([x1, x2, x3])
		x_u = cnn_block(x123, cnn_unit=cnn_concat_unit, kernel_size=1)
		
		# x9 = cnn_block(input_x, cnn_unit=cnn_unit, kernel_size=7)
		# x8 = cnn_block(x9, cnn_unit=cnn_unit, kernel_size=5)
		# x98 = layers.Concatenate()([x9, x8])
		# x7 = cnn_block(x98, cnn_unit=cnn_unit, kernel_size=3)
		# x987 = layers.Concatenate()([x9, x8, x7])
		# x_l = cnn_block(x98, cnn_unit=cnn_concat_unit, kernel_size=1)

		# x_f = cnn_block(input_x, cnn_unit=cnn_unit, kernel_size=3)
		# x_input_f = cnn_block(input_x, cnn_unit=cnn_unit, kernel_size=1)
		# # x_f = layers.Concatenate()([x_input_f, x_f])
		# # x_f = cnn_block(x_f, cnn_unit=cnn_concat_unit, kernel_size=1)

		x9 = cnn_block(input_x, cnn_unit=cnn_unit, kernel_size=3)
		x8 = cnn_block(x9, cnn_unit=cnn_unit, kernel_size=3)
		x98 = layers.Concatenate()([x9, x8])
		x7 = cnn_block(x98, cnn_unit=cnn_unit, kernel_size=3)
		x987 = layers.Concatenate()([x9, x8, x7])
		x_l = cnn_block(x987, cnn_unit=cnn_concat_unit, kernel_size=1)

		x = layers.Concatenate()([x_u, x_l, input_x])
		x = cnn_block(x, cnn_unit=cnn_concat_unit, kernel_size=1)
		x = layers.SpatialDropout1D(rate=dropout_rate)(x)
		if bilstm:
			for i in range(bilstm-1):
				x = Bidirectional(pLSTM(bilstm_unit))(x)
				x = layers.SpatialDropout1D(rate=dropout_rate)(x)
			x = Bidirectional(pLSTM(bilstm_unit, return_sequences=False))(x)
			x = layers.Dropout(rate=dropout_rate)(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		model.summary()
		return model

	return dense_nn