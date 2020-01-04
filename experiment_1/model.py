from keras import models
from keras import layers
from keras import backend as K
from keras.layers import Activation, BatchNormalization 

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def nn_model(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform')))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_1(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(32, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(32,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.3))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.LSTM(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_2(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.LSTM(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_3(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(32, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(32,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(32,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.5))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.LSTM(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_4(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform')))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(256, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(512, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(512, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_5(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform')))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_6(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform')))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_7(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(64,kernel_initializer = 'he_uniform')))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_8(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.4))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform')))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model

def nn_model_9(input_shape_1,input_shape_2):

	model = models.Sequential()
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128, kernel_initializer = 'he_uniform', return_sequences=True, input_shape=(input_shape_1,input_shape_2))))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform', return_sequences=True)))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Bidirectional(layers.CuDNNLSTM(128,kernel_initializer = 'he_uniform')))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(1024, activation='relu', kernel_initializer = 'he_uniform'))
	model.add(layers.Dropout(rate=0.7))
	model.add(layers.Dense(23, activation='linear', kernel_initializer = 'he_uniform'))
	return model