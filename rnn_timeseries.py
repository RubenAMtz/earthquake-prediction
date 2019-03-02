import tensorflow as tf
import numpy as np
import pandas as pd
import glob
import matplotlib.pyplot as plt
import random
import tqdm
BASE_DIR = 'D:/kaggle/LANL_Earthquake prediction/'
train_segments_list = glob.glob(BASE_DIR + 'train_segments/*.csv')

sample_size = 500
sorted_sample = [
    train_segments_list[i] for i in sorted(random.sample(range(len(train_segments_list)), sample_size))
]
segments = pd.DataFrame(columns=['acoustic_data','time_to_failure'])
for file in tqdm.tqdm(sorted_sample):
    segment = pd.read_csv(file, index_col=0)
    segments = segments.append(segment, ignore_index=True)
audio = segment['acoustic_data'].values
ttf = segment['time_to_failure'].values

#print(audio.shape)
#prepare the data so that each 20 instances are listed together:
def shape_data(data = None):
    counter = 0
    elements = []
    groups = []
    for element in data:
        counter += 1
        elements.append([element])
        if counter == 20:
            counter = 0
            groups.append(elements)
            elements = []
    return groups

audio_ = shape_data(audio)
ttf_ = shape_data(ttf)

n_steps = 20
n_inputs = 1
n_neurons = 100
n_outputs = 1


X = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_steps, n_outputs])
cell = tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu)

# wrapper to allow for a single output
cell = tf.contrib.rnn.OutputProjectionWrapper(
tf.contrib.rnn.BasicRNNCell(num_units=n_neurons, activation=tf.nn.relu),
output_size=n_outputs)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

learning_rate = 0.001
loss = tf.reduce_mean(tf.square(outputs - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
training_op = optimizer.minimize(loss)
init = tf.global_variables_initializer()

n_iterations = 5000
batch_size = 50
with tf.Session() as sess:
    init.run()
    for iteration in range(n_iterations):
        #X_batch, y_batch = [...] # fetch the next training batch
        X_batch = audio_
        y_batch = ttf_
        sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        if iteration % 100 == 0:
            mse = loss.eval(feed_dict={X: X_batch, y: y_batch})
            print(iteration, "\tMSE:", mse)

    train_segments_list = glob.glob(BASE_DIR + 'train_segments/*.csv')
    segment = pd.read_csv(train_segments_list[1])
    audio = segment['acoustic_data'].values
    ttf = segment['time_to_failure'].values

    X_new = shape_data(audio)

    y_pred = sess.run(outputs, feed_dict={X: X_new})
    print(y_pred.shape)
    predictions = []
    for instance in y_pred:
        for time_step in instance:
            predictions.append(time_step[0])
        
    plt.plot(predictions)
    plt.plot(ttf)
    plt.show()