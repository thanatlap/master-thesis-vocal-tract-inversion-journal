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


def init_resnet():

	def residual_block(input_x):
		
		x = pConv1D(64, kernel_size=3)(input_x)
		x = BatchNormalization()(x)
		x = Activation('elu')(x)
		x = pConv1D(64, kernel_size=3)(x)
		x = BatchNormalization()(x)
		x = keras.layers.Add()([x, input_x])
		output = Activation('elu')(x)
		return output

	def resnet(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x = pConv1D(64, kernel_size=7)(input_x)
		x = BatchNormalization()(x)
		x = Activation('elu')(x)
		x = residual_block(x)
		x = residual_block(x)
		x = residual_block(x)
		x = layers.GlobalAveragePooling1D()(x)
		x = Flatten()(x)
		outputs = pDense(N_OUTPUTS, activation='linear')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		return model
	return resnet

# def init_cnn_bilstm():

# 	def residual_block(input_x):
		
# 		x = pConv1D(64, kernel_size=3)(input_x)
# 		x = BatchNormalization()(x)
# 		x = Activation('elu')(x)
# 		x = pConv1D(64, kernel_size=3)(x)
# 		x = BatchNormalization()(x)
# 		x = keras.layers.Add()([x, input_x])
# 		output = Activation('elu')(x)
# 		return output

# 	def cnn_bilstm(input_shape_1,input_shape_2):

# 		input_x = InputLayer(input_shape=(input_shape_1,input_shape_2))
# 		x = pConv1D(64, kernel_size=7)(input_x)
# 		x = BatchNormalization()(x)
# 		x = Activation('elu')(x)
# 		x = residual_block(x)
# 		x = residual_block(x)
# 		x = residual_block(x)
# 		x = layers.GlobalAveragePooling1D()(x)
# 		x = Flatten()(x)
# 		output = pDense(N_OUTPUTS, activation='linear')(x)

# 	return cnn_bilstm


# def init_res():

# 	def residual_layer():



# 		return residual

# 	def res(input_shape_1,input_shape_2):

# 		input_x = InputLayer(input_shape=(input_shape_1,input_shape_2))


# 		return model
# 	return res


