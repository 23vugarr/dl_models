import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('star_classification.csv')
target = np.array(pd.get_dummies(df['class']).astype('int64'))
features = StandardScaler().fit_transform(np.array(df.drop(columns='class')))

xtrain, xtest, ytrain, ytest = train_test_split(features, target, random_state=2, test_size=0.3)
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import SGD, Adam, RMSprop, Adagrad
from tensorflow.keras import regularizers

model = Sequential()

model.add(Dense(16, activation='relu', input_shape = (xtrain.shape[1], )))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(256, activation='relu', kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_regularizer='l2'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(3, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])
train = model.fit(
    xtrain, 
    ytrain,
    batch_size=32, 
    verbose=1,
    epochs=10,
    validation_data= (xtest, ytest)
)
# show test data accuracy
score = model.evaluate(xtest, ytest, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

