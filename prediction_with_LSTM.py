import glob
import pandas as pd
import numpy as np
from sklearn.utils import validation
"""
appended_data = []

for f in glob.glob("./*.csv"):
    df = pd.read_csv(f, header = 0)
    appended_data.append(df)

df = pd.concat(appended_data)
df.to_csv("appended.csv")
"""
df = pd.read_csv("appended.csv")
print(df.head())

dataset = pd.DataFrame(index=range(0,len(df)), columns = ["Temp1"])
for i in range(len(df)):
    dataset["Temp1"][i] = df["Temp1"][i]




# split a univariate sequence into samples
def split_sequence(sequence, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequence)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the sequence
		if out_end_ix > len(sequence):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequence[i:end_ix], sequence[end_ix:out_end_ix]
		X.append(seq_x)
		y.append(seq_y)
	return np.array(X), np.array(y)

data = dataset.values

nstepsin, nstepsout = 5, 5

X , y = split_sequence(data, nstepsin, nstepsout)

print(X.shape, y.shape)

X = X.reshape(X.shape[0], X.shape[1]*X.shape[2])
y = y.reshape(y.shape[0], y.shape[1]*y.shape[2])
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

X_train, X_test = X[:-30000],X[-30000:] 
y_train, y_test = y[:-30000],y[-30000:] 

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
y_train = y_train.reshape(y_train.shape[0], y_train.shape[1], 1)

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from keras import optimizers

model = Sequential()

model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(50, activation="relu"))
model.add(Dense(5))


from keras.callbacks import ModelCheckpoint, EarlyStopping
filepath = "models/{epoch:02d}-{loss:.4f}-{val_loss:.4f}-{mse:.4f}-{val_mse:.4f}.hdf5"

callbacks = [EarlyStopping(monitor="val_loss", patience=50),
             ModelCheckpoint(filepath, monitor="loss", save_best_only=True, mode="min")]


#optimizers.adam(lr=0.001)

model.compile(optimizer="adam", loss="mse", metrics=["mse"])
#model.fit(X_train, y_train, validation_split=0.2, epochs=100, callbacks=callbacks, batch_size=8)



model.load_weights("./models/05-0.0001-0.0001-0.0001-0.0001.hdf5")

import time

for i in range(0, X_test.shape[0]):
    results_summary = []
    X_input = X_test[i,:]
    print(X_input.shape)
    X_input = X_input.reshape(1,X_input.shape[0],1)
    print(X_input.shape)
    X_input = model.predict(X_input)
    print(X_input.shape)
    forecast = scaler.inverse_transform(X_input)


    y_input = y_test[i,:]
    y_input = y_input.reshape(y_input.shape[0],1).T
    print(y_input.shape)
    actual = scaler.inverse_transform(y_input)

    results_summary.append(actual)
    results_summary.extend(forecast)


    df_animate = pd.DataFrame(results_summary)
    df_animate = df_animate.T
    df_animate.to_csv("real time temp.csv", mode="a", header=False, index=False)

    print(results_summary)
    time.sleep(0.5)
    