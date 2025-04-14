import pandas as pd
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import warnings
warnings.simplefilter('ignore', FutureWarning)

def regression_model():
    model = Sequential()
    model.add(Input(shape=(n_cols,)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

filepath='https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/DL0101EN/labs/data/concrete_data.csv'
concrete_data = pd.read_csv(filepath)

predictors = concrete_data[concrete_data.columns[concrete_data.columns != 'Strength']]
target = concrete_data['Strength']
# predictors.head()
# target.head()
#normalizing
predictors_norm = (predictors - predictors.mean()) / predictors.std()
n_cols = predictors_norm.shape[1]

model = regression_model()
model.fit(predictors_norm, target, validation_split=0.3, epochs=100, verbose=2)