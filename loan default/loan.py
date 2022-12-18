import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('Loan_Default.csv')
df = df.drop(columns=['ID', 'year'])

for i in df.columns:
    if(df[i].dtype not in ['object', 'str']):
        df[i].fillna(df[i].mean(), inplace=True)
        
df.dropna(inplace=True)
for i in df.columns:
    if(df[i].dtype in ['object', 'str'] and i != 'Status'):
        df = pd.concat([pd.get_dummies(df[i]).astype('int64'), df.drop(columns=i)], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

target = np.array(df['Status'])
features = StandardScaler().fit_transform(np.array(df.drop(columns='Status')))

xtrain, xtest, ytrain, ytest = train_test_split(features, target, test_size=0.2, random_state=20)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam
model = Sequential()

model.add(Dense(64, activation='relu', input_shape=(xtrain.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(512, activation='relu', kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(256, activation='relu',  kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(64, activation='relu',  kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer=Adam(lr=0.001), loss='binary_crossentropy', metrics=['accuracy'])
train = model.fit(
    xtrain, ytrain,
    epochs=10,
    batch_size=32,
    validation_data=(xtest, ytest)
)
model.evaluate(xtest, ytest)
