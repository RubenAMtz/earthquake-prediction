# READ AUDIO DATA
# CONVERT IT TO REQUIRED SHAPE (RESHAPE)
# CONFIGURE NETWORK FOR 1D CONV
import numpy as np
np.random.seed(1337)  # for reproducibility
import pandas as pd

from sklearn import preprocessing 
from sklearn import metrics
from sklearn.model_selection import train_test_split

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, Conv1D, Input
from keras.utils import np_utils
from keras.utils.vis_utils import model_to_dot
import h5py
from keras.models import load_model
import matplotlib.pyplot as plt
import os
import tqdm

def exodial(path=None, size=1500):
    if path != None:
        files = os.listdir(path)
        samples = pd.DataFrame(columns=['acoustic_data','time_to_failure'])
        for file in tqdm.tqdm(files):
            segment = pd.read_csv(path + '/'+ file)
            segment = segment.drop(segment.columns[0], axis=1)
            sample = segment.sample(n=size)
            samples = pd.concat([samples, sample], ignore_index=True)
        return samples
    return None

def preprocess_data(filename):
    data = pd.read_csv(filename)
    y = data['time_to_failure']
    X = data["acoustic_data"]
    # Use a range scaling to scale all variables to between 0 and 1
    X = X.values.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler((-1,1))
    X = min_max_scaler.fit_transform(X) # Watch out for putting back in columns here

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0, shuffle=True)
    
    y_train = y_train.values
    y_test = y_test.values
    #X_train = X_train.reshape(X_train.shape[0], 1, 1)
    #X_test = X_test.reshape(X_test.shape[0], 1, 1)
    
    return X_train, X_test, y_train, y_test

# START:
#data = exodial('./train_segments/', size=1500)
data = pd.read_csv('./train_segments/train_segment_0000.csv')
#data = pd.read_csv('./example.csv')
y = data['time_to_failure']
X = data["acoustic_data"]

# Use a range scaling to scale all variables to between 0 and 1
X = X.values.reshape(-1,1)
min_max_scaler = preprocessing.MinMaxScaler((0,1))
X = min_max_scaler.fit_transform(X) # Watch out for putting back in columns here

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=0, shuffle=True)

# convert to numpy arrays
y_train = y_train.values
y_test = y_test.values

#plt.plot(list(range(0, 500)), X_train[:500])
#plt.plot(list(range(0, 500)), y_train[:500])
#plt.plot(list(range(0, 500)), X_test[:500])
#plt.plot(list(range(0, 500)), y_test[:500])
#plt.show()

#X_train = X_train.reshape(X_train.shape[0], 1, 1)
#X_test = X_test.reshape(X_test.shape[0], 1, 1)

X_train = X_train.reshape(X_train.shape[0],1, 1)
#y_train = y_train.reshape(1, y_train.shape[0])
X_test = X_test.reshape(X_test.shape[0],1, 1)
print(X_test.shape)

input_shape = (1, 1)
#input_shape = (X_train.shape[1], 1)
#X_train = X_train.astype('float32')
#X_test = X_test.astype('float32')
# print(y_train.shape)
# print(y_test.shape)

#x = np.linspace(0, 10, 1000)
#y = 10*np.sin(x)
#x = x.reshape(len(x), 1)
#y = y.reshape(len(y), 1)
#print(x.shape)

if not os.path.isfile('deep_model.h5'):
    # Set up model
    model = Sequential()
    model.add(Conv1D(filters=1, kernel_size=500, padding="same", input_shape=input_shape, activation='sigmoid'))
    model.add(Conv1D(filters=64, kernel_size=500, padding="same", activation='sigmoid'))
    #model.add(Conv1D(filters=128, kernel_size=500, padding="same", activation='sigmoid'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units= 64, activation='sigmoid'))
    model.add(Dropout(0.5))
    model.add(Dense(units = 128, activation='sigmoid'))
    model.add(Dense(units = 128, activation='sigmoid'))
    model.add(Dense(units = 64, activation='sigmoid'))
    model.add(Dense(units = 1))
    
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='mse', optimizer=adam, metrics=['mae'])
    model.fit(X_train, y_train, batch_size=32, epochs=1)

    model.save('deep_model.h5')
    keras.utils.print_summary(model)
else:
    model = load_model('deep_model.h5')
    keras.utils.print_summary(model)

#score = model.evaluate(X_test, y_test, verbose=1)
#print('Test loss:', score[0])
#print('Test mse:', score[1])

# Read another file
# X_test, y_test = preprocess_data('./train_segments/train_segment_0099.csv')
#X_test = np.linspace(11, 21, 1000)
#X_test = X_test.reshape(len(X_test),1)
#y_test = np.sin(x)
#y_test = y_test.reshape(len(y_test),1)
score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score)

y_predict = model.predict(X_test, verbose=1)
#print(y_predict.shape)
#print(y_test.shape)
#print(y_predict[0:20,0])

# if not os.path.isfile('predictions.csv'):
#     y_pred_pd = pd.DataFrame(columns =['y_predict','y_test'])
#     y_pred_pd['y_predict'] = y_predict[:,0]
#     y_pred_pd['y_test'] = y_test
#     y_pred_pd.to_csv('predictions.csv')

plt.plot( y_predict, '-b', label="y_predict")
plt.plot( y_test, '--r', label="y_test")
plt.legend()
plt.show()
