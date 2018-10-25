
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

energydata_complete = energydata_complete.drop(['Appliances'], axis=1)

Y = energydata_complete.iloc[:, 0].values

labelencoder_Y_1 = LabelEncoder()
Y_labled = labelencoder_Y_1.fit_transform(Y)

onehotencoder = OneHotEncoder(categorical_features = [0])
Y_onehot = onehotencoder.fit_transform(Y_labled.reshape(Y_labled.shape[0], 1)).toarray()
print(Y_onehot)

Y_columnnames = []
for i in range(Y_onehot.shape[1]):
    if i < 10:
        Y_columnnames.append("Y1_0{0}".format(i))
    else:
        Y_columnnames.append("Y1_{0}".format(i))

Y_onehot = pd.DataFrame(Y_onehot, columns = Y_columnnames)

sc = MinMaxScaler(feature_range = (0, 1))
variables_scaled = pd.DataFrame(sc.fit_transform(energydata_complete.iloc[:, 1:]), columns = list(energydata_complete.iloc[:, 1:].columns))
print(variables_scaled.head())

# separate the data into training set and testing set
X_train_scaled = variables_scaled.iloc[:round(np.shape(variables_scaled)[0] * 0.8), :]
Y_train_encoded = Y_onehot.iloc[:round(np.shape(Y_onehot)[0] * 0.8), :]

X_test_scaled = variables_scaled.iloc[round(np.shape(variables_scaled)[0] * 0.8):, :]
Y_test_encoded = Y_onehot.iloc[round(np.shape(Y_onehot)[0] * 0.8):, :]

X_train_scaled = np.reshape(X_train_scaled.values, (X_train_scaled.values.shape[0], X_train_scaled.values.shape[1], 1))
X_test_scaled = np.reshape(X_test_scaled.values, (X_test_scaled.values.shape[0], X_test_scaled.values.shape[1], 1))

# Initialising the RNN
layers = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
layers.add(LSTM(units = 16, return_sequences = True, input_shape = (X_train_scaled.shape[1], 1)))
layers.add(Dropout(0.2))

layers.add(Flatten())

# Adding the output layer
layers.add(Dense(Y_train_encoded.shape[1], activation='softmax'))

layers.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# all the following code will be executed manually for every iteration
layers.fit(X_train_scaled, Y_train_encoded, epochs = 100, batch_size = 512)

# layers.save('layers2_2.h5')  # creates a HDF5 file 'my_model.h5'
# layers = load_model('layers2_2.h5')

predicted_energy = layers.predict(X_test_scaled)
accuracy = sum(np.argmax(predicted_energy, axis = 1) == np.argmax(Y_test_encoded.values, axis = 1)) / len(Y_test_encoded.values)
print("The accuracy(categorical) is: {0}".format(accuracy))

predicted_energy_inversed = labelencoder_Y_1.inverse_transform(np.argmax(predicted_energy, axis = 1))
predicted_mse = mse(predicted_energy_inversed, Y[round(np.shape(Y)[0] * 0.8):])
print("The mean squared error is: {0}".format(predicted_mse))

plt.plot(Y[round(np.shape(Y)[0] * 0.8):], color = 'red', label = 'real energy')
plt.plot(predicted_energy_inversed, color = 'blue', label = 'Predicted energy')
plt.title('Appliances energy prediction')
plt.xlabel('Time')
plt.ylabel('energy')
plt.legend()
plt.show()
