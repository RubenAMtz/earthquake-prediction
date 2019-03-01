import os
import glob
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
#from multiprocessing import Pool
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
from joblib import load, dump
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.utils.vis_utils import model_to_dot
import h5py
from keras.models import load_model

# train_file = pd.read_csv('D:/kaggle/LANL_Earthquake prediction/train/train.csv')
# sample = train_file.sample(n=int(len(train)*0.1), random_state=42)

BASE_DIR = 'D:/kaggle/LANL_Earthquake prediction/'

def create_train_XY(X=None, y=None):
    for id, file in enumerate(tqdm(train_segments_list)):
        segment = pd.read_csv(file)
        if X is not None :
            create_features(id, segment, X)
        if y is not None:
            y.loc[id, 'time_to_failure'] = segment['time_to_failure'].values.mean()
    if X is not None:
        X.to_csv(BASE_DIR + 'train_X.csv')
    if y is not None:
        y.to_csv(BASE_DIR + 'train_y.csv')
        
def create_segments():
    for idx, chunk in enumerate(pd.read_csv(BASE_DIR + 'train/train.csv', chunksize=150000)):
        chunk_name = "train_segment_{:04d}.csv".format(idx)
        chunk.to_csv(BASE_DIR + 'train_segments/'+ chunk_name)

def plot_sample_file(path=BASE_DIR, filename=None):
    segment = pd.read_csv(path + filename)
    audio_col = segment['acoustic_data']
    ttf_col = segment['time_to_failure']
    title = '"Acoustic data" and "Time to Failure" in {}'.format(filename)
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(audio_col, color='r')
    ax1.set_ylabel('acoustic data', color='r')
    plt.legend(['acoustic data'], loc=(0.01, 0.95))
    plt.plot(audio_col, color='r')
    ax2 = ax1.twinx()
    plt.plot(ttf_col, color='b')
    ax2.set_ylabel('time to failure', color='b')
    plt.legend(['time to failure'], loc=(0.01, 0.9))
    plt.grid(True)
    plt.show()

def create_test_X(test_X):
    for id in tqdm(test_X.index):
        segment = pd.read_csv(BASE_DIR + 'test/' + id + '.csv')
        create_features(id, segment, test_X)

    test_X.to_csv(BASE_DIR + 'test_X.csv')

def scale(dataframe):
    scaler = StandardScaler()
    scaler.fit(dataframe)
    scaled_dataframe = pd.DataFrame(scaler.transform(dataframe), columns=dataframe.columns)
    return scaled_dataframe
    
# utility method for feature creation
def add_trend_feature(arr, abs_values=False):
    idx = np.array(range(len(arr)))
    if abs_values:
        arr = np.abs(arr)
    lr = LinearRegression()
    lr.fit(idx.reshape(-1, 1), arr)
    return lr.coef_[0]

def create_features(id, segment, train_X):
    # pd.values returns a numpy representation of the data
    # pd.Series creates a one-dimensional ndarray with axis labels (including time series)
    x_ts = pd.Series(segment['acoustic_data'].values)
    zc = np.fft.fft(x_ts)
    
    train_X.loc[id, 'mean'] = x_ts.mean()
    train_X.loc[id, 'std'] = x_ts.std()
    train_X.loc[id, 'max'] = x_ts.max()
    train_X.loc[id, 'min'] = x_ts.min()

    #FFT transform values
    realFFT = np.real(zc)
    imagFFT = np.imag(zc)
    train_X.loc[id, 'Rmean'] = realFFT.mean()
    train_X.loc[id, 'Rstd'] = realFFT.std()
    train_X.loc[id, 'Rmax'] = realFFT.max()
    train_X.loc[id, 'Rmin'] = realFFT.min()
    train_X.loc[id, 'Imean'] = imagFFT.mean()
    train_X.loc[id, 'Istd'] = imagFFT.std()
    train_X.loc[id, 'Imax'] = imagFFT.max()
    train_X.loc[id, 'Imin'] = imagFFT.min()
    train_X.loc[id, 'Rmean_last_5000'] = realFFT[-5000:].mean()
    train_X.loc[id, 'Rstd__last_5000'] = realFFT[-5000:].std()
    train_X.loc[id, 'Rmax_last_5000'] = realFFT[-5000:].max()
    train_X.loc[id, 'Rmin_last_5000'] = realFFT[-5000:].min()
    train_X.loc[id, 'Rmean_last_15000'] = realFFT[-15000:].mean()
    train_X.loc[id, 'Rstd_last_15000'] = realFFT[-15000:].std()
    train_X.loc[id, 'Rmax_last_15000'] = realFFT[-15000:].max()
    train_X.loc[id, 'Rmin_last_15000'] = realFFT[-15000:].min()

    train_X.loc[id, 'mean_change_abs'] = np.mean(np.diff(x_ts))
    train_X.loc[id, 'mean_change_rate'] = np.mean(np.nonzero((np.diff(x_ts) / x_ts[:-1]))[0])
    train_X.loc[id, 'abs_max'] = np.abs(x_ts).max()
    train_X.loc[id, 'abs_min'] = np.abs(x_ts).min()
    
    train_X.loc[id, 'std_first_50000'] = x_ts[:50000].std()
    train_X.loc[id, 'std_last_50000'] = x_ts[-50000:].std()
    train_X.loc[id, 'std_first_10000'] = x_ts[:10000].std()
    train_X.loc[id, 'std_last_10000'] = x_ts[-10000:].std()
    
    train_X.loc[id, 'avg_first_50000'] = x_ts[:50000].mean()
    train_X.loc[id, 'avg_last_50000'] = x_ts[-50000:].mean()
    train_X.loc[id, 'avg_first_10000'] = x_ts[:10000].mean()
    train_X.loc[id, 'avg_last_10000'] = x_ts[-10000:].mean()
    
    train_X.loc[id, 'min_first_50000'] = x_ts[:50000].min()
    train_X.loc[id, 'min_last_50000'] = x_ts[-50000:].min()
    train_X.loc[id, 'min_first_10000'] = x_ts[:10000].min()
    train_X.loc[id, 'min_last_10000'] = x_ts[-10000:].min()
    
    train_X.loc[id, 'max_first_50000'] = x_ts[:50000].max()
    train_X.loc[id, 'max_last_50000'] = x_ts[-50000:].max()
    train_X.loc[id, 'max_first_10000'] = x_ts[:10000].max()
    train_X.loc[id, 'max_last_10000'] = x_ts[-10000:].max()
    
    train_X.loc[id, 'max_to_min'] = x_ts.max() / np.abs(x_ts.min())
    train_X.loc[id, 'max_to_min_diff'] = x_ts.max() - np.abs(x_ts.min())
    train_X.loc[id, 'count_big'] = len(x_ts[np.abs(x_ts) > 500])
    train_X.loc[id, 'sum'] = x_ts.sum()
    
    # Mean over indices were change rate is different than zero (Change Rate, averaged (Difference over actual quantity))
    train_X.loc[id, 'mean_change_rate_first_50000'] = np.mean(np.nonzero((np.diff(x_ts[:50000]) / x_ts[:50000][:-1]))[0])
    train_X.loc[id, 'mean_change_rate_last_50000'] = np.mean(np.nonzero((np.diff(x_ts[-50000:]) / x_ts[-50000:][:-1]))[0])
    train_X.loc[id, 'mean_change_rate_first_10000'] = np.mean(np.nonzero((np.diff(x_ts[:10000]) / x_ts[:10000][:-1]))[0])
    train_X.loc[id, 'mean_change_rate_last_10000'] = np.mean(np.nonzero((np.diff(x_ts[-10000:]) / x_ts[-10000:][:-1]))[0])
    
    train_X.loc[id, 'q95'] = np.quantile(x_ts, 0.95)
    train_X.loc[id, 'q99'] = np.quantile(x_ts, 0.99)
    train_X.loc[id, 'q05'] = np.quantile(x_ts, 0.05)
    train_X.loc[id, 'q01'] = np.quantile(x_ts, 0.01)
    
    train_X.loc[id, 'abs_q95'] = np.quantile(np.abs(x_ts), 0.95)
    train_X.loc[id, 'abs_q99'] = np.quantile(np.abs(x_ts), 0.99)
    train_X.loc[id, 'abs_q05'] = np.quantile(np.abs(x_ts), 0.05)
    train_X.loc[id, 'abs_q01'] = np.quantile(np.abs(x_ts), 0.01)

    train_X.loc[id, 'mad'] = x_ts.mad()
    train_X.loc[id, 'kurt'] = x_ts.kurtosis()
    train_X.loc[id, 'skew'] = x_ts.skew()
    train_X.loc[id, 'med'] = x_ts.median()

    train_X.loc[id, 'Moving_average_700_mean'] = x_ts.rolling(window=700).mean().mean(skipna=True)
    train_X.loc[id, 'Moving_average_1500_mean'] = x_ts.rolling(window=1500).mean().mean(skipna=True)
    train_X.loc[id, 'Moving_average_3000_mean'] = x_ts.rolling(window=3000).mean().mean(skipna=True)
    train_X.loc[id, 'Moving_average_6000_mean'] = x_ts.rolling(window=6000).mean().mean(skipna=True)
    
    # Exponential MAVE
    ewma = pd.Series.ewm
    train_X.loc[id, 'exp_Moving_average_300_mean'] = (ewma(x_ts, span=300).mean()).mean(skipna=True)
    train_X.loc[id, 'exp_Moving_average_3000_mean'] = ewma(x_ts, span=3000).mean().mean(skipna=True)
    train_X.loc[id, 'exp_Moving_average_30000_mean'] = ewma(x_ts, span=30000).mean().mean(skipna=True)

    train_X.loc[id, 'iqr'] = np.subtract(*np.percentile(x_ts, [75, 25]))
    train_X.loc[id, 'q999'] = np.quantile(x_ts,0.999)
    train_X.loc[id, 'q001'] = np.quantile(x_ts,0.001)
    train_X.loc[id, 'ave10'] = stats.trim_mean(x_ts, 0.1)

    # trend features
    train_X.loc[id, 'trend'] = add_trend_feature(x_ts.values)
    train_X.loc[id, 'abs_trend'] = add_trend_feature(x_ts.values, abs_values=True)

    for windows in [10, 100, 1000]:
        x_roll_std = x_ts.rolling(windows).std().dropna().values
        x_roll_mean = x_ts.rolling(windows).mean().dropna().values
        
        train_X.loc[id, 'ave_roll_std_' + str(windows)] = x_roll_std.mean()
        train_X.loc[id, 'std_roll_std_' + str(windows)] = x_roll_std.std()
        train_X.loc[id, 'max_roll_std_' + str(windows)] = x_roll_std.max()
        train_X.loc[id, 'min_roll_std_' + str(windows)] = x_roll_std.min()
        train_X.loc[id, 'q01_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.01)
        train_X.loc[id, 'q05_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.05)
        train_X.loc[id, 'q95_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.95)
        train_X.loc[id, 'q99_roll_std_' + str(windows)] = np.quantile(x_roll_std, 0.99)
        train_X.loc[id, 'av_change_abs_roll_std_' + str(windows)] = np.mean(np.diff(x_roll_std))
        train_X.loc[id, 'av_change_rate_roll_std_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_std) / x_roll_std[:-1]))[0])
        train_X.loc[id, 'abs_max_roll_std_' + str(windows)] = np.abs(x_roll_std).max()
        
        train_X.loc[id, 'ave_roll_mean_' + str(windows)] = x_roll_mean.mean()
        train_X.loc[id, 'std_roll_mean_' + str(windows)] = x_roll_mean.std()
        train_X.loc[id, 'max_roll_mean_' + str(windows)] = x_roll_mean.max()
        train_X.loc[id, 'min_roll_mean_' + str(windows)] = x_roll_mean.min()
        train_X.loc[id, 'q01_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.01)
        train_X.loc[id, 'q05_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.05)
        train_X.loc[id, 'q95_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.95)
        train_X.loc[id, 'q99_roll_mean_' + str(windows)] = np.quantile(x_roll_mean, 0.99)
        train_X.loc[id, 'av_change_abs_roll_mean_' + str(windows)] = np.mean(np.diff(x_roll_mean))
        train_X.loc[id, 'av_change_rate_roll_mean_' + str(windows)] = np.mean(np.nonzero((np.diff(x_roll_mean) / x_roll_mean[:-1]))[0])
        train_X.loc[id, 'abs_max_roll_mean_' + str(windows)] = np.abs(x_roll_mean).max()
    # NEW FEATURES
    # Spectral Shape Features:
        #Centroid (center of gravity)
        #Roll off (the frequency which corresponds to the 85% of the total energy from the spectrum)
    
    #
    
if __name__ == "__main__":
    
    if os.path.isfile(BASE_DIR + 'train_X.csv'):
        train_X = pd.read_csv(BASE_DIR + 'train_X.csv', index_col=0)
    if os.path.isfile(BASE_DIR + 'train_y.csv'):
        train_y = pd.read_csv(BASE_DIR + 'train_y.csv', index_col=0)

    train_segments_list = glob.glob(BASE_DIR + 'train_segments/*.csv')
    
    if not os.path.isfile(BASE_DIR + 'train_X.csv'):
        train_X = pd.DataFrame(index=range(len(train_segments_list)), dtype=np.float64)
        create_train_XY(X=train_X)

    if not os.path.isfile(BASE_DIR + 'train_y.csv'):
        train_y = pd.DataFrame(index=range(len(train_segments_list)), dtype=np.float64, columns=['time_to_failure'])
        create_train_XY(y=train_y)
    
    scaled_train_X = scale(train_X)

    if not os.path.isfile(BASE_DIR + 'test_X.csv'):
        submission = pd.read_csv(BASE_DIR + 'sample_submission.csv', index_col='seg_id')
        # configures test_X
        test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)
        create_test_X(test_X)
    else:
        test_X = pd.read_csv(BASE_DIR + 'test_X.csv', index_col=0)
    
    scaled_test_X = scale(test_X)

    ## CONFIGURE OUR MODEL
    # Apparently feature selection did not good at all. We will now try with all the features

    X_train_, X_val, y_train_, y_val = train_test_split(scaled_train_X, train_y, test_size=0.33, random_state=42)

    print(X_train_.shape)

    if not os.path.isfile('deep_model.h5'):
        # Set up model
        model = Sequential()
        model.add(Dense(X_train_.shape[1], input_shape=(X_train_.shape[1],), activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(80, activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(60,  activation='sigmoid'))
        model.add(Dropout(0.5))
        model.add(Dense(30, activation='sigmoid'))
        model.add(Dropout(0.5))
        #model.add(Dense(10, activation='sigmoid'))
        #model.add(Dropout(0.5))
        model.add(Dense(1))
        
        sgd = keras.optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='mse', optimizer=sgd, metrics=['mae'])
        history = model.fit(X_train_, y_train_, batch_size=32, epochs=2000, validation_data=[X_val, y_val])

        model.save('deep_model.h5')
        keras.utils.print_summary(model)
    else:
        model = load_model('deep_model.h5')
        keras.utils.print_summary(model)

    score = model.evaluate(X_val, y_val, verbose=0)
    print('Test loss:', score)

    y_predict = model.predict(X_val, verbose=1)

    mae_eval = metrics.mean_absolute_error(y_val, y_predict)

    # PREDICTION USING TRAINING DATA:
    #y_predict_train = gbregressor.predict(X_train_)
    y_predict_train = model.predict(X_train_)
    mae_train = metrics.mean_absolute_error(y_train_, y_predict_train)

    print("Mean Absolute Error(train) " +  str(mae_train))
    print("Mean Absolute Error(eval): " +  str(mae_eval))

    print("History keys: ", history.history.keys())

    title = "Y_predict vs Y_Validation" 
    fig, ax1 = plt.subplots(figsize=(12, 8))
    plt.title(title)
    plt.plot(y_val.values, color='r')
    ax1.set_ylabel('y_validation', color='r')
    plt.legend(['prediction'], loc=(0.01, 0.95))
    plt.plot(y_predict, color='b')
    ax2 = ax1.twinx()
    ax2.set_ylabel('y_prediction', color='b')
    plt.legend(['groundT'], loc=(0.01, 0.9))
    plt.grid(True)
    plt.show()

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # PREDICTION USING TEST DATA:
    #y_predict = gbregressor.predict(scaled_test_X)
    if not os.path.isfile(BASE_DIR + 'final_submission.csv'):
        y_predict = model.predict(scaled_test_X)
        submission = pd.read_csv('sample_submission.csv')
        submission['time_to_failure'] = y_predict
        submission.to_csv('final_submission.csv', index=False)