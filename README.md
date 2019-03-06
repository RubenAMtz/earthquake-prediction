# LANL Earthquake Prediction

Can you predict upcoming laboratory earthquakes?

#### Kaggle Competition: [Link](https://www.kaggle.com/c/LANL-Earthquake-Prediction)

# Public Results

Team name: ***RubenAMtz***

| Commit       | Score (mae) | Date     |
| ------------ | ----------- | -------- |
| First        | 1.808       | 13/02/19 |
| Second       | 1.724       | 28/02/19 |
| Third        | 1.765       | 01/03/19 |
| Fourth       | 1.787       | 01/03/19 |
| Fifth        | 1.665       | 02/03/19 |


![Earthquake](https://cdn.newsapi.com.au/image/v1/36aab19faa109c662cc4361696831a64?width=1024 "Earthquake detection")


# Data structure

- columns = 2
- column names: acoustic_data, time_to_failure
- acoustic_data:  the seismic signal [int16]
- time_to_failure: the time (in seconds) until the next laboratory earthquake [float64]

# File descriptions

### Train data:  
- format: .csv
- 600M instances in training set

### Test data:  
- format: .csv
- test segments = 2624 (files)
- 150k instances in every test segment

*** 

# Strategy

Step 1:

- Split inputs from outputs

Step 2:

- Split the training set in segments (with as many instances as the test sets)
- 150K instances in test sets, so... 600M / 150K ~ 4K segments out of the training set
- Create as many training segments needed and store in files

Step 3:

- Feature engineering (create features from inputs) to TRAINING data and TEST data


![Audio features](https://s3-ap-south-1.amazonaws.com/av-blog-media/wp-content/uploads/2017/08/23233229/audio-features.png "Audio features")


- Apply x methods to generate features for every TRAINING segment

    Number of features created: ~~71~~ 131

As our model is trained based on the newly created features, we need to create the same features for our test set, which is the
data that will be used to make predictions:

- Apply same methods to generate same features for every TEST segment
- Concatenate generated features in a single TEST set.
- Scale the data (train and test)
- Save files in folders train_features and test_features

Step 4:

- Configure the model: ensemble { GradientBoostingRegression }
- As we don't have a VAL set, we need to split the TRAIN set, to create a Validation set. Split so that we have 66% train vs 33% validation:
  
| Set      | Inputs   | Output |
| -------- |:--------:| ------:|
| TRAIN    | Yes      | Yes    |
| VAL      | Yes      | Yes    |
| TEST     | Yes      | No     |
  
- Define evaluation metrics, 'mean average error' as defined by competition rules.
- Evaluate model with 5-fold cross-validation
- Train and predict
- Submit file

***
# Updates 06/03/19

    - CNN will be trained using STFT plots from each training segment.

# Updates 02/03/19

- Features added:

    - Peak counting based on mean and std from local files (and statistics)
    - MFCC for last and first samples per local file (and statistics)
    - Removed some features
- Optimized GradientBoostingRegressor by GridSearchCV:

    - Smaller learning rate
    - Increased the number of estimators

# Updates 01/03/19

- More features were added
- GridSearch improved our model in ~0.1%
- A deep dense model was implemented
- Dense model seems to fit better the training data but overfits.



