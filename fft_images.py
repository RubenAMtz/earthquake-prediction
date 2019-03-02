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
import librosa
import librosa.display
import os

BASE_DIR = 'D:/kaggle/LANL_Earthquake prediction/'

train_segments_list = glob.glob(BASE_DIR + 'train_segments/*.csv')
ttfailures = []
files_names = []
maxs = []
for id, file in enumerate(tqdm(train_segments_list)):
    #reading each file
    if not os.path.isfile('./img_segments/' + file[-22:][:-4] + '.png'):
        segment = pd.read_csv(file)
        
        # extract audio signal
        audio_signal = segment['acoustic_data'].values
        # stft
        D = np.abs(librosa.stft(y=np.float32(audio_signal), n_fft=1000))
        """
        total time	            0.0375	s
        time per sample	    0.00000025	s
        frequency captured	   4000000	Hz
                
        bins	                   150	
        samples per bin	          1000	
        window time size	   0.00025  s
        frequency resolution      4000	Hz
        """
        #librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), y_axis='log', x_axis='time')
        # librosa.display.specshow(librosa.amplitude_to_db(D), y_axis='log', x_axis='time')
        # plt.title('Power spectrogram')
        # plt.colorbar(format='%+2.0f dB')
        # plt.tight_layout()
        # plt.show()
        librosa.display.specshow(librosa.amplitude_to_db(D))
        
        plt.savefig('./img_segments/' + file[-22:][:-4] + '.png', bbox_inches='tight', pad_inches=0.0)
        
        # max time to failure per file
        ttfailure = segment['time_to_failure'].values.max()

        files_names.append(file[-22:][:-4])
        ttfailures.append(ttfailure)

df = pd.DataFrame(files_names, columns=['segment'])
df['time_to_failure'] = ttfailures

df.to_csv('image-time.csv')
