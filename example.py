import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split

# Generate dummy data
# x_train = np.random.random((1000, 20))
# y_train = np.random.randint(2, size=(1000, 1))
# x_test = np.random.random((100, 20))
# y_test = np.random.randint(2, size=(100, 1))
x = np.linspace(0, 1, 10000)
y = - 1 *np.sin(x)
#
#y_train = y_train.reshape(-1, 1)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, train_size=0.7, random_state=0, shuffle=True)
x_train = x_train.reshape(-1, 1)
x_test = x_test.reshape(-1, 1)
#y_test = y_test.reshape(-1, 1)

print(x_train.shape)
print(x_test.shape)
# print(x_train)

model = Sequential()
model.add(Dense(64, input_dim=1, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(32, activation='sigmoid'))
model.add(Dropout(0.5))
model.add(Dense(16, activation='tanh'))
model.add(Dropout(0.5))
model.add(Dense(1))

adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer=adam, metrics=['mean_squared_error'])

model.fit(x_train, y_train,
        epochs=320,
        batch_size=128,
        verbose=1)

score = model.evaluate(x_test, y_test, batch_size=128)
print(score)
    
y_predict = model.predict(x_test, verbose=1)

plt.plot( y_predict, '-b', label="y_predict")
plt.plot( y_test, '--r', label="y_test")
plt.legend()
plt.show()

# x = np.arange(200).reshape(-1,1) / 50
# y = np.sin(x)

# model = Sequential([
#     Dense(40, input_shape=(1,), activation='sigmoid'),
#     Dense(12, activation='sigmoid'),
#     Dense(1)
#     ])

# model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

# for i in range(40):
#     model.fit(x, y, nb_epoch=25, batch_size=8, verbose=0)
#     predictions = model.predict(x)
#     print(np.mean(np.square(predictions - y)))

