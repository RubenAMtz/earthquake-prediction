"""
* Explore data (600 M)
    - Sample
    - Visualize
    - Size
    - Summary

* Data structure:
    Data:
    - columns = 2
    - column names: audio, time_to_failure

    - input: audio (int)
    - output: time_to_failure (float)

    Train data:
    - format: .csv
    - 600M instances in training set

    Test data:
        - format: .csv
        - Test segments = 2626 (files)
        - 150k instances in every test set

-----------------------------------------------------

Step 1:

* Split inputs from outputs:

    This means we have to isolate the *audio* column from the *time_to_failure* column

Step 2:

* Split the training set in segments (with as many instances as the test sets)
    - 150K instances in test sets, so... 600M / 150K ~ 4K segments out of the training set
    - Create as many training segments needed and store in files

Step 3:

* Feature engineering (create features from inputs) to TRAINING data and TEST data
    - Apply any method to generate features for every TRAINING segment

    As our model is trained based on the newly created features, we need to create the same features for our test set, which is the
    data that will be used to make predictions:

    - Apply same methods to generate same features for every TEST segment
    - Scale the data (train and test)
    - Save files in folders train_features and test_features

Step 4:

* Configure the model (ensemble)

* As we only have train file (inputs-output relationship), we need to split it, to create a Validation set.
    - Split so that we have 66% train vs 33% validation
    {
    Train:
        X = inputs, Y= output
    Test (X for sample sumbission):
        X = inputs, Y= NA
    Sample submission:
        Y = ? to be compared vs ground truth
    }

* Define evaluation metrics = 'mean average error'

* Define feature importance

* Train model and evaluate with k-fold cross-validation

* Submit file

"""
import gc
import os
import time
import glob
import logging
import datetime
import warnings
import numpy as np
import pandas as pd
# import xgboost as xgb
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.svm import NuSVR, SVR
from sklearn.kernel_ridge import KernelRidge
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold,StratifiedKFold, RepeatedKFold, cross_val_score
from multiprocessing import Pool
warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn import ensemble
from joblib import load, dump

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
    
    ewma = pd.Series.ewm
    train_X.loc[id, 'exp_Moving_average_300_mean'] = (ewma(x_ts, span=300).mean()).mean(skipna=True)
    train_X.loc[id, 'exp_Moving_average_3000_mean'] = ewma(x_ts, span=3000).mean().mean(skipna=True)
    train_X.loc[id, 'exp_Moving_average_30000_mean'] = ewma(x_ts, span=6000).mean().mean(skipna=True)

    train_X.loc[id, 'iqr'] = np.subtract(*np.percentile(x_ts, [75, 25]))
    train_X.loc[id, 'q999'] = np.quantile(x_ts,0.999)
    train_X.loc[id, 'q001'] = np.quantile(x_ts,0.001)
    train_X.loc[id, 'ave10'] = stats.trim_mean(x_ts, 0.1)
    
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

    # just once
    
    scaled_train_X = scale(train_X)
    #scaled_train_Y = scale(train_y)
    #print(scaled_train_Y.head(10))

    if not os.path.isfile(BASE_DIR + 'test_X.csv'):
        submission = pd.read_csv(BASE_DIR + 'sample_submission.csv', index_col='seg_id')
        # configures test_X
        test_X = pd.DataFrame(columns=train_X.columns, dtype=np.float64, index=submission.index)
        create_test_X(test_X)
    else:
        test_X = pd.read_csv(BASE_DIR + 'test_X.csv', index_col=0)
    
    scaled_test_X = scale(test_X)

    # FEATURE SELECTION
        # Using wrappers
        # Configure a randomeforest ensemble classifier:

        # algoritmo de busqueda
            # generar combinaciones de variables
                # 100
        # algoritmo de classificacion/regression

    if not os.path.isfile(BASE_DIR + 'rfr_model.joblib'):
        feat_select_model = ensemble.RandomForestRegressor(criterion="mae", max_depth=50, min_samples_split=50, n_estimators = 500, verbose=1000, n_jobs=5)
        feat_select_model = feat_select_model.fit(scaled_train_X, train_y)
        dump(feat_select_model, 'rfr_model.joblib')
    else:
        feat_select_model = load('rfr_model.joblib')
    
    #print(feat_select_model.feature_importances_)
    model = SelectFromModel(feat_select_model, prefit=True)

    # FEATURES SELECTED:
    new_scaled_X = model.transform(scaled_train_X)
    new_scaled_X_test = model.transform(scaled_test_X)
    print(new_scaled_X.shape)
    print(new_scaled_X_test.shape)
    # print(model.get_support(indices=True))
    # print(train_X.iloc[0,model.get_support(indices=True)])
    print(train_X.columns[model.get_support(indices=True)])
    
    ## CONFIGURE OUR MODEL
    # Apparently feature selection did not good at all. We will now try with all the features
    #X_train_, X_val, y_train_, y_val = train_test_split(scaled_train_X, train_y, test_size=0.33, random_state=42)
    X_train_, X_val, y_train_, y_val = train_test_split(scaled_train_X, train_y, test_size=0.33, random_state=42)

    if not os.path.isfile(BASE_DIR + 'gbregressor_model.joblib'):
        gbregressor = ensemble.GradientBoostingRegressor(loss='ls', n_estimators=1000, criterion="mae", learning_rate=0.001, n_iter_no_change= 200)
        scores = cross_val_score(gbregressor, X_train_, y_train_, cv=5, n_jobs = 5, verbose=1000, scoring = 'neg_mean_absolute_error')
        print(scores)
        gbregressor.fit(X_train_,y_train_)

        dump(gbregressor, 'gbregressor_model.joblib')
    else:
        gbregressor = load('gbregressor_model.joblib')

    # PREDICTION USING VALIDATION DATA
    y_predict = gbregressor.predict(X_val)

    mae_eval = metrics.mean_absolute_error(y_val, y_predict)

    # PREDICTION USING TRAINING DATA:
    y_predict_train = gbregressor.predict(X_train_)
    mae_train = metrics.mean_absolute_error(y_train_, y_predict_train)

    print("Mean Absolute Error(train) " +  str(mae_train))
    print("Mean Absolute Error(eval): " +  str(mae_eval))

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

    # PREDICTION USING TEST DATA:
    y_predict = gbregressor.predict(scaled_test_X)
    submission = pd.read_csv('sample_submission.csv')
    submission['time_to_failure'] = y_predict
    submission.to_csv('final_submission.csv', index=False)








    # -------------------------------

    # print(dir(X_ts))
    
    # segment = pd.read_csv(train_segments_list[0])
    # acoustic = segment['acoustic_data'].values
    # index = list(range(len(acoustic)))
    # x = [[i, element] for i, element in enumerate(acoustic)]
    # # print(x)
    # #plt.scatter(index, acoustic)
    # #plt.show()
    # kmeans = KMeans(n_clusters=40, random_state=0).fit(x)
    # plt.scatter(index, acoustic, c=kmeans.labels_, s=1)
    # plt.show()