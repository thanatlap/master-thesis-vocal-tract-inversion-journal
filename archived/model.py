import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape, GRU, InputLayer, AlphaDropout, Activation, BatchNormalization, Dropout, Flatten, Dense, Bidirectional, LSTM, Conv1D, SpatialDropout1D, Concatenate, Multiply, Add
from functools import partial

N_OUTPUTS = 17

pDense = partial(Dense, kernel_initializer='he_uniform', activation='relu')
pLSTM = partial(LSTM, kernel_initializer='he_uniform', return_sequences=True)
pConv1D = partial(Conv1D, padding = 'same', activation = 'linear', kernel_initializer = 'he_uniform')

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
		model.summary()
		return model
	return baseline

def init_FCNN(unit_ff = 1024, layer_num = 5, drop_rate=0.5):

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

def init_bilstm(unit=128, bi_layer_num=5, drop_rate=0.5):

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

def init_LTRCNN():

	def LTRCNN(input_shape_1,input_shape_2):
		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x1 = pConv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(input_x)
		x2 = pConv1D(128, kernel_size=3, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x1)
		x = layers.Concatenate()([x1, x2])
		x = pLSTM(1024, return_sequences=False)(x)
		x = pDense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0005))(x)
		outputs = pDense(N_OUTPUTS, activation='linear', kernel_regularizer=regularizers.l2(0.0005))(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		model.summary()
		return model
	return LTRCNN

def init_senet(feature_layer, bilstm, se_enable, cnn_unit, bilstm_unit, dropout_rate,
	first_kernel=7, res_kernel=3, reduction_ratio = 2, activation_fn='relu', embedded_path = None):

	if embedded_path:
		embedded = tf.keras.models.load_model(embedded_path)
		embedded_layers = tf.keras.Sequential()
		for layer in embedded.layers[:-2]:
			layer.trainable = False
			embedded_layers.add(layer)
		embedded_layers.summary()


	def cnn_block(input_x, cnn_unit, kernel_size):
		x = pConv1D(cnn_unit, kernel_size=kernel_size)(input_x)
		x = BatchNormalization()(x)
		x = Activation(activation_fn)(x)
		return x
	
	def residual_block(input_x):
		x = pConv1D(cnn_unit, kernel_size=res_kernel)(input_x)
		x = BatchNormalization()(x)
		x = Activation(activation_fn)(x)
		x = pConv1D(cnn_unit, kernel_size=5)(x)
		return x

	def se_block(input_x):
		x = layers.GlobalAveragePooling1D()(input_x)
		channel_shape = getattr(x, '_shape_val')[-1]
		x = Reshape((1, channel_shape))(x)
		x = Dense(channel_shape // reduction_ratio, activation=activation_fn, kernel_initializer='he_uniform')(x)
		x = Dense(channel_shape, activation='sigmoid', kernel_initializer='he_uniform')(x)
		x = Multiply()([x, input_x])
		return x

	def se_res_block(input_x):
		if se_enable:
			se_x = se_block(input_x)
			re_x = residual_block(se_x)
		else:
			re_x = residual_block(input_x)
		x = Add()([re_x, input_x])
		x = BatchNormalization()(x)
		output = Activation(activation_fn)(x)
		return x

	def senet_nn(input_shape_1,input_shape_2):

		input_x = keras.Input(shape=(input_shape_1,input_shape_2))
		x = cnn_block(input_x, cnn_unit, first_kernel)
		for i in range(feature_layer):
			x = se_res_block(x)
		if dropout_rate: x = Dropout(rate=dropout_rate)(x)
		if embedded_path:
			embedded = embedded_layers(input_x)
			x = Concatenate()([x, embedded])
		# add conv and concat
		x = cnn_block(input_x, 39, 1)
		x = Concatenate()([x, input_x])
		for i in range(bilstm-1):
			x = Bidirectional(pLSTM(bilstm_unit))(x)
			if dropout_rate: x = Dropout(rate=dropout_rate)(x)
		x = Bidirectional(pLSTM(bilstm_unit, return_sequences=False))(x)
		if dropout_rate: x = Dropout(rate=dropout_rate)(x)
		outputs = Dense(N_OUTPUTS, activation='linear', kernel_initializer='he_uniform')(x)
		model = keras.Model(inputs=input_x, outputs=outputs)
		model.summary()
		return model

	return senet_nn