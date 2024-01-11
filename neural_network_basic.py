# import pandas as pd
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Dropout
from tensorflow.python.keras.layers import ReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import *
from tensorflow.python.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
import numpy as np
import tensorflow.keras as tfk
import tensorflow as tf
from final_layer_weights import final_layer_weights
# from sklearn.preprocessing import StandardScaler
from scaler import *

class pf_to_pinj(tfk.layers.Layer):
	def __init__(self, nodes, input_dim, weights1):
		super(pf_to_pinj, self).__init__()
		self.w = self.add_weight(
			name = 'w',
			shape = (input_dim, nodes),
			initializer = 'random_normal',
			trainable = False,
		)

		self.b = self.add_weight(
			name = 'b',
			shape = (nodes,),
			initializer = 'zeros',
			trainable = False
		)
		self.nodes = nodes
		self.input_dim = input_dim
		self.weights1 = weights1

	def call(self, inputs):
		self.W = self.weights1
		return tf.matmul(inputs, self.W) + self.b




class Deep_Network():
	"""docstring for ClassName"""
	def __init__(self, layers=3, nodes=118, lr = 1e-4, epoch =100, batch_norm = False, dropout = 0):
		self.layers = layers
		self.nodes = nodes
		self.lr = lr
		self.epoch = epoch
		self.batch_norm = batch_norm
		self.dropout = dropout
		self.model = Sequential()
		self.save = False
		self.y_normalise = False
		self.static_layer = False

		self.last_layer_weights = final_layer_weights()
		self.last_layer_weights1 = tf.convert_to_tensor(self.last_layer_weights)
		self.last_layer_weights2 = tf.cast(self.last_layer_weights1, tf.float32)
		self.output = "powers"
		self.loss = "mse"



	def model_make1(self, X, y):
		self.model.add(Dense(self.nodes, activation='relu', input_dim=X.shape[1]))
		for i in range(self.layers-1):
			self.model.add(Dense(self.nodes, activation = 'relu'))
		if self.batch_norm:
			self.model.add(BatchNormalization())
		# model.add(ReLU())
		self.model.add(Dropout(self.dropout))
		self.model.add(Dense(y.shape[1]))

	def model_make(self, X, y):
		n_last = self.last_layer_weights.shape[1]
		n_second_last = self.last_layer_weights.shape[0]

		# self.model.add(Dense(self.nodes, activation='relu', input_dim=X.shape[1]))
		for i in range(self.layers):
			if i==0:
				self.model.add(Dense(self.nodes, activation='relu', input_dim=X.shape[1]))
			elif i==self.layers-1:

				self.model.add(pf_to_pinj(n_last, n_second_last, self.last_layer_weights))
			elif i ==self.layers-2:
				self.model.add(Dense(n_second_last))
			else:
				self.model.add(Dense(self.nodes, activation = 'relu', kernel_initializer='he_normal'))
		# if self.batch_norm:
			# self.model.add(BatchNormalization())
		# model.add(ReLU())
		# self.model.add(Dropout(self.dropout))

	def custom_loss(self, x, xhat):
		y = tf.linalg.matmul(x, self.last_layer_weights2)
		yhat = tf.linalg.matmul(xhat, self.last_layer_weights2)
		return tf.math.reduce_mean((y-yhat)**2)


	def model_compile(self):
		# optimizer = Adam(lr = 1e-3)
		# optimizer = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
		if self.loss == "mse":
			self.model.compile(loss="MeanSquaredError", optimizer = 'adam', metrics=[tfk.metrics.MeanAbsoluteError()])  # noqa: E501
		elif self.loss == "custom":
			self.model.compile(loss=self.custom_loss, optimizer = 'adam', metrics=[tfk.metrics.MeanAbsoluteError()])  # noqa: E501
		# self.model.summary()


	def test_error(self, x_test, y_test):

		yhat = self.model.predict(x_test)
		# l, b = yhat.shape

		# no_load_buses = [5, 9, 30, 37, 38, 63, 64, 68, 71, 81]
		# no_load_index = [i-1 for i in no_load_buses]

		if self.y_normalise:
			yhat_a = self.y_scaler.inverse_transform(yhat)
			ytest_a = self.y_scaler.inverse_transform(y_test)
		else:
			yhat_a = yhat
			ytest_a = y_test

		if self.save:
			np.savetxt('yhat.csv', yhat_a, delimiter = ',')
			# np.savetxt('ytest.csv', ytest_a, delimiter = ',')

		
		mse = [0 for i in range(ytest_a.shape[1])]
		for i in range(ytest_a.shape[1]):
			mse[i] = mean_squared_error(yhat_a[:,i], ytest_a[:,i])
	
		mape = [0 for i in range(ytest_a.shape[1])]
		for i in range(ytest_a.shape[1]):
			mape[i] = mean_absolute_percentage_error(yhat_a[:,i], ytest_a[:,i])

		r2 = [0 for i in range(ytest_a.shape[1])]
		for i in range(ytest_a.shape[1]):
			r2[i] = r2_score(ytest_a[:,i], yhat_a[:,i])

		# mse2 = np.matmul(mse, self.last_layer_weights)
		return mse, mape,  r2

	def split_normalise(self, X, y, Pinj, Qinj):
		self.x_scaler = scaler(X.shape[1])
		self.y_scaler = scaler(y.shape[1])
		x_train, x_test, y_train, y_test, _, Ptest, _, Qtest = train_test_split(X, y, Pinj, Qinj, test_size=0.2)
		x_training, x_val, y_training, y_val = train_test_split(x_train, y_train, test_size=0.1)

		if self.save:
			np.savetxt('ptest.csv', Ptest, delimiter = ',')
			np.savetxt('qtest.csv', Qtest, delimiter = ',')


		x_train_scaled = self.x_scaler.fit_transform(x_training)
		x_val_scaled = self.x_scaler.transform(x_val)
		x_test_scaled = self.x_scaler.transform(x_test)


		if self.y_normalise:
			y_train_scaled = self.y_scaler.fit_transform(y_training)
			y_val_scaled = self.y_scaler.transform(y_val)
			y_test_scaled = self.y_scaler.transform(y_test)
		else:
			y_train_scaled = y_training
			y_val_scaled = y_val
			y_test_scaled = y_test

		return x_train_scaled, x_val_scaled, x_test_scaled, y_train_scaled, y_val_scaled, y_test_scaled


	def model_parse(self, X, y, Pinj, Qinj, ntest=100):
		mse = [0 for i in range(ntest)]
		mape = [0 for i in range(ntest)]
		r2 = [0 for i in range(ntest)]
		for i in range(ntest):
			if self.static_layer:
				self.last_layer_weights = final_layer_weights()
				self.model_make(X, y)
			else:
				self.model_make1(X, y)
			self.model_compile()

			X_training, X_val, X_test, y_training, y_val, y_test = self.split_normalise(X, y, Pinj, Qinj)

			x_training = np.nan_to_num(X_training)
			x_val = np.nan_to_num(X_val)
			x_test = np.nan_to_num(X_test)

			Y_training = np.nan_to_num(y_training)
			Y_val = np.nan_to_num(y_val)
			Y_test = np.nan_to_num(y_test)


			# early_stopping = EarlyStopping(monitor='val_loss', patience=30)
			# filepath="temp/weightsbest"
			# checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, mode='min', save_best_only=True, save_weights_only=False)

			reduce_lr = ReduceLROnPlateau(monitor='loss',mode="min", factor=0.2,patience=10, min_lr=0.0001)
			
			self.model.fit(x_training, Y_training, validation_data=(x_val, Y_val),
					 verbose=0, epochs=self.epoch, batch_size=64, shuffle=True,
					 callbacks=[reduce_lr])

			mse[i], mape[i], r2[i] = self.test_error(x_test, Y_test)

		return mse, mape, r2


