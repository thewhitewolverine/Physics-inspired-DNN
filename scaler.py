import numpy as np
class scaler():
	def __init__(self, size):
		self.size = size
		self.std = [0 for i in range(size)]
		self.mean = [0 for i in range(size)]

	def fit(self, x):
		for i in range(self.size):
			self.std[i] = np.std(x[:, i])
			self.mean[i] = np.mean(x[:, i])

	def transform(self, x):
		xn = np.copy(x)
		for i in range(self.size):
			xn[:, i] = (x[:, i]-self.mean[i])/self.std[i]
		return xn

	def inverse_transform(self, x):
		xn = np.copy(x)
		for i in range(self.size):
			xn[:, i] = x[:, i] * self.std[i] + self.mean[i]

		return xn

	def fit_transform(self, x):
		xn = np.copy(x)
		for i in range(self.size):
			self.std[i] = np.std(x[:, i])
			self.mean[i] = np.mean(x[:, i])
			xn[:, i] = (x[:, i]-self.mean[i])/self.std[i]

		return xn

	def fit_transform_output(self, x):
		xn = np.copy(x)
		for i in range(self.size):
			self.std[i] = np.std(x[:, i])
			self.mean[i] = np.mean(x[:, i])
			xn[:, i] = (x[:, i]/self.mean[i])

		return xn

	def transform_output(self, x):
		xn = np.copy(x)
		for i in range(self.size):
			xn[:, i] = (x[:, i]/self.mean[i])
		return xn

	def inverse_transform_output(self, x):
		xn = np.copy(x)
		for i in range(self.size):
			xn[:, i] = x[:, i] * self.mean[i]

		return xn