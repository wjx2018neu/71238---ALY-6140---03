
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random as rn
import tensorflow as tf

from keras import backend as K
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.models import load_model
from keras.layers import LSTM
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

def mse(ar1, ar2):
    return ((ar1 - ar2) ** 2).mean()

def rmse(ar1, ar2):
    return np.sqrt(mse(ar1, ar2))

# make the results reproducible
os.environ["PYTHONHASHSEED"] = '0'
np.random.seed(1)
rn.seed(2)
tf.set_random_seed(3)
session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

# loading and preprocessing data
energydata_complete = pd.read_csv("energydata_complete.csv").iloc[:, 1:]

energydata_complete = energydata_complete.drop(['lights'], axis=1)

sc = MinMaxScaler(feature_range = (0, 1))

values_scaled = pd.DataFrame(sc.fit_transform(energydata_complete))

# separate the data into training set and testing set
X_train_scaled = values_scaled.iloc[:round(np.shape(values_scaled)[0] * 0.8), 1:]
Y_train_scaled = values_scaled.iloc[:round(np.shape(values_scaled)[0] * 0.8), 0]

X_test_scaled = values_scaled.iloc[round(np.shape(values_scaled)[0] * 0.8):, 1:]
Y_test_scaled = values_scaled.iloc[round(np.shape(values_scaled)[0] * 0.8):, 0]
Y_test = energydata_complete.iloc[round(np.shape(energydata_complete)[0] * 0.8):, 0].values

X_train_scaled = np.reshape(X_train_scaled.values, (X_train_scaled.values.shape[0], X_train_scaled.values.shape[1], 1))
X_test_scaled = np.reshape(X_test_scaled.values, (X_test_scaled.values.shape[0], X_test_scaled.values.shape[1], 1))

# Initialising the RNN
layers = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
layers.add(LSTM(units = 16, return_sequences = True, input_shape = (X_train_scaled.shape[1], 1)))
layers.add(Dropout(0.2))

layers.add(Flatten())

# Adding the output layer
layers.add(Dense(1))

layers.compile(optimizer = 'adam', loss = 'mean_squared_error')

# all the following code will be executed manually for every iteration
layers.fit(X_train_scaled, Y_train_scaled, epochs = 100, batch_size = 512)

# layers.save('layers1_2.h5')  # creates a HDF5 file 'my_model.h5'
# layers = load_model('layers1_2.h5')

predicted_energy = layers.predict(X_test_scaled)

predicted_inversed = sc.inverse_transform(np.hstack((predicted_energy, np.zeros((predicted_energy.shape[0], np.shape(energydata_complete)[1] - 1)))))[:, 0]
predicted_mse = mse(predicted_inversed, Y_test)
print("The mean squared error is: {0}".format(predicted_mse))

plt.plot(Y_test, color = 'red', label = 'real energy')
plt.plot(predicted_inversed, color = 'blue', label = 'Predicted energy')
plt.title('Appliances energy prediction')
plt.xlabel('Time')
plt.ylabel('energy')
plt.legend()
plt.show()
