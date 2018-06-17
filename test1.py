from numpy import concatenate
import numpy as np
from matplotlib import pyplot
from pandas import concat, DataFrame, read_excel
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.layers.noise import GaussianNoise
from keras.callbacks import TensorBoard


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg

# read data
dataset = read_excel('1v_cosd_turb_a.xlsx', header=0)
dataset = dataset.dropna(axis=0, how='all')
values = dataset.values
values = values.astype('float32')

# normalize features
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(values)

# reshape data
n_lookback = 20
n_features = 4
n_pre_term = 1
reframed = series_to_supervised(scaled, n_lookback, n_pre_term)
values = reframed.values

# divide sets
n_train_terms = int(reframed.shape[0]/2)
train = values[:n_train_terms, :]
test = values[n_train_terms:, :]

# divide input, output, and set output of predicting timestep to 0
train_X = train.copy()
train_X[:, -1] = 0
test_X = test.copy()
test_X[:, -1] = 0
train_y = train[:, -1]
train_y = np.matrix(train_y)
train_y = np.transpose(train_y)
test_y = test[:, -1]
test_y = np.matrix(test_y)
test_y = np.transpose(test_y)

# reshape input into LSTM target shape
train_X = train_X.reshape((train_X.shape[0], n_lookback+1, n_features))
test_X = test_X.reshape((test_X.shape[0], n_lookback+1, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

# design network
model = Sequential()
model.add(GaussianNoise(stddev=0.005))
model.add((LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]))))
model.add(Dense(64))
model.add(Dense(16))
model.add(Dense(8))
model.add(Dense(4))
model.add(Dense(1))
model.summary()
model.compile(loss='mse', optimizer='adam')

# train network
history = model.fit(train_X, train_y, epochs=100, verbose=2, shuffle=False, validation_data=(test_X, test_y),
					callbacks=[TensorBoard(log_dir='./graph')])

# plot history
los1 = np.divide(history.history['loss'], np.square(np.mean(train_y)))
los2 = np.divide(history.history['val_loss'], np.square(np.mean(test_y)))
pyplot.plot(los1, label='train')
pyplot.plot(los2, label='test')
pyplot.xlabel('epochs')
pyplot.ylabel('loss')
pyplot.ylim(ymin=0)
pyplot.legend()
pyplot.show()

# predict train set
y0 = model.predict(train_X)

# invert scaling for forecast
temp = np.zeros((train.shape[0], 3))
y0 = concatenate((temp, y0), axis=1)
y0 = scaler.inverse_transform(y0)
y0 = y0[:, -1:]

# invert scaling for actual
inv_ty = concatenate((temp, train_y), axis=1)
inv_ty = scaler.inverse_transform(inv_ty)
inv_ty = inv_ty[:, -1:]

# predict train set
yhat = model.predict(test_X)

# invert scaling for forecast
temp = np.zeros((test.shape[0], 3))
inv_yhat = concatenate((temp, yhat), axis=1)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, -1]

# invert scaling for actual
inv_y = concatenate((temp, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]

# draw result
pyplot.figure('result')
pyplot.subplot(211)
pyplot.plot(inv_ty, 'r', label='train_real')
pyplot.plot(y0[:, 0], 'b', label='train_prediction')
pyplot.xlabel('time/10s')
pyplot.ylabel('acceleration/(m/s²)')
pyplot.legend()
pyplot.subplot(212)
pyplot.plot(inv_y, 'r', label='test_real')
pyplot.plot(inv_yhat, 'b', label='test_prediction')
pyplot.xlabel('time/10s')
pyplot.ylabel('acceleration/(m/s²)')
pyplot.legend()
pyplot.show()
