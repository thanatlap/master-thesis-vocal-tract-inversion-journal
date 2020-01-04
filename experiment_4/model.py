import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Activation, BatchNormalization 

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

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
	model.add(layers.Dense(18, activation='linear', kernel_initializer = 'he_uniform'))
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
	model.add(layers.Dense(18, activation='linear', kernel_initializer = 'he_uniform'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))

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
	model.add(layers.Dense(18, activation='linear', kernel_initializer = 'he_uniform'))
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
	model.add(layers.Dense(18))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
	return model

def nn_bilstm_21(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Bidirectional(layers.LSTM(128, return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.LSTM(128)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(1024, activation='relu' ))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Dense(18, activation='linear'))
	return model

def nn_fc_bilstm_fc(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
	return model

def nn_fc_bilstm_fc_2(input_shape_1,input_shape_2):

	model = tf.keras.Sequential()
	model.add(layers.Dense(512, activation = 'elu', input_shape=(input_shape_1,input_shape_2)))
	model.add(layers.Dense(512, activation = 'elu'))
	model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True)))
	model.add(layers.Bidirectional(layers.LSTM(256)))
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
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
	model.add(layers.Dense(18, activation='linear'))
	return model