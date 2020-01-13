import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, BatchNormalization 

import config as cf

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


def custom_loss(y_true, y_pred):
	r2 = R2(y_true, y_pred)
	mse = tf.keras.losses.MSE(y_true,y_pred)
	return (1-r2)*mse + mse

def custom_loss2(y_true, y_pred):
	r2 = R2(y_true, y_pred)
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	return (1-r2)*mse + mse

def custom_loss3(y_true, y_pred):
	r2 = R2(y_true, y_pred)
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	return (1-r2)*K.sqrt(mse) + mse

def custom_loss4(y_true, y_pred):
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	adjustR2 = AdjustR2(y_true, y_pred)
	return (1-adjustR2)*mse + mse

def custom_loss5(y_true, y_pred):
	mse = K.mean(K.square(y_pred - y_true), axis=-1)
	adjustR2 = AdjustR2(y_true, y_pred)
	return (1-adjustR2)*K.sqrt(mse) + mse

def custom_loss6(y_true, y_pred):
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
	model.add(layers.Dense(16, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_fc_bn(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	
	model.add(layers.Flatten(input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dense(1024, kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(layers.Dense(1024, kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(layers.Dense(1024, kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(layers.Dense(1024, kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(layers.Dense(1024, kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(layers.Dense(16, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_bilstm(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_2(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()

	forward_layer = layers.LSTM(128, return_sequences=True)
	backward_layer = layers.LSTM(128, activation='relu', return_sequences=True,go_backwards=True)

	model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(forward_layer, backward_layer=backward_layer))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(16, activation='linear'))

	model.summary()

	return model

def nn_lstm(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.LSTM(128, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.LSTM(256,kernel_initializer = 'he_uniform', return_sequences=True))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.LSTM(256,kernel_initializer = 'he_uniform', return_sequences=True))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.LSTM(256,kernel_initializer = 'he_uniform', return_sequences=True))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.LSTM(256,kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(16, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_cnn(input_shape_1,input_shape_2):
	model = tf.keras.Sequential()
	model.add(layers.Conv2D(16,(3,3), padding='same', kernel_initializer = 'he_uniform', input_shape=(input_shape_1,input_shape_2,1)))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv2D(16,(3,3), padding='same', kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv2D(16,(3,3), padding='same', kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(32,(3,3), padding='same', kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv2D(32,(3,3), padding='same',kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv2D(32,(3,3), padding='same',kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.MaxPooling2D())
	model.add(layers.Conv2D(64,(3,3), padding='same',kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv2D(64,(3,3), padding='same',kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv2D(64,(3,3), padding='same',kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Flatten())
	model.add(layers.Dense(1024, kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Dense(1024, kernel_initializer = 'he_uniform'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Dense(16))
	return model

def nn_bilstm_3(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_4(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_5(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(64)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_6(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_7(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_8(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_9(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_10(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_11(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(BatchNormalization())
	model.add(Activation('relu'))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_12(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_13(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_14(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_15(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_16(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_17(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_18(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_19(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_20(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_21(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fc_bilstm_fc(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fc_bilstm_fc_drop(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fc_bilstm_fc_2(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fc_bilstm_fc_2_drop(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fc_bilstm_cn_fc_drop(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(128, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_3(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(128, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_4(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(128, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_5(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(128, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_6(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(128, 5, activation='relu', padding='same'))
	model.add(layers.Conv1D(128, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model


# --- 
# model nn_fbc_7 8 are too big for this hardware
# model nn_fbc_9 10 are similar to 7 and 8 but adding pooling layer
# model nn_fbc_11 testing the effect of pooling layer comparing to model nn_fc_bilstm_cn_fc_drop
def nn_fbc_7(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_8(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_9(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.MaxPool1D())
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_10(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.MaxPool1D())
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.MaxPool1D())
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_11(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.MaxPool1D())
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_12(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.MaxPool1D())
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_22(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_23(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_23(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(1024, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_24(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bilstm_25(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_fbc_13(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(1024, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation = 'elu'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.MaxPool1D())
	model.add(layers.Conv1D(64, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_cbf_1(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Conv1D(128, 3, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_cbf_2(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Conv1D(128, 3, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_cbf_3(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(64, 5, padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv1D(128, 3, padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(256, activation='linear')))
	model.add(BatchNormalization())
	model.add(Activation('tanh'))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_cbf_4(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Conv1D(128, 3, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(1024, activation = 'elu'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_cbf_5(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Conv1D(128, 3, activation='elu', padding='same'))
	model.add(layers.MaxPool1D())
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bcf_1(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bcf_2(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(1024, activation='elu'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bcf_3(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
	model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(1024, activation='elu'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bcf_4(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(64, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
	model.add(layers.Conv1D(32, 5, activation='relu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(1024, activation='elu'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model


def nn_bcf_5(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	return model

def nn_bcf_6(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(128, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_bcf_7(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(64, 5, activation='elu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_bcf_8(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_bcf_9(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 3, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 3, activation='elu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_cn_1(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.MaxPool1D())
	model.add(layers.Conv1D(64, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 3, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.MaxPool1D())
	model.add(layers.Conv1D(128, 3, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(128, 3, activation='elu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_cn_2(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(32, 3, padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv1D(32, 3, padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.MaxPool1D())
	model.add(layers.Conv1D(64, 3, padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv1D(64, 3, padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.MaxPool1D())
	model.add(layers.Conv1D(128, 3, padding='same'))
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Conv1D(128, 3, padding='same'))
	model.add(layers.Flatten())
	model.add(BatchNormalization())
	model.add(Activation('elu'))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_bcf_10(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 3, activation='elu', padding='same'))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 3, activation='elu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_bcf_11(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Conv1D(64, 3, activation='elu', padding='same'))
	model.add(layers.Flatten())
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_bcf_12(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Conv1D(32, 3, activation='elu', padding='same', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	model.summary()
	return model

def nn_bilstm_22(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='elu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(16, activation='linear'))
	return model