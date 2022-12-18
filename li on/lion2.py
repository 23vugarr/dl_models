import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

df = pd.read_csv('lithium-ion batteries.csv')

for i in df.columns:
    print(i, df[i].nunique())
df['Has Bandstructure'] = np.where(df['Has Bandstructure'] == True, 1, 0)
df['Nsites'] = df['Nsites'].astype('float64')

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dersigmoid(x):
    return x * (1.0 - x)

def loss_func(x, y):
    return (x - y) ** 2

def relu(x):
    return np.maximum(0,x)

def derrelu(x):
    return np.where(x>0, 1, 0)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

def dersoftmax(x):
    return softmax(x) * (1 - softmax(x))

features = df.drop(columns=['Materials Id', 'Formula', 'Spacegroup', 'Crystal System'], axis=1)
target = np.array(pd.get_dummies(df['Crystal System']).astype('float64'))

features = np.array(features)
features = StandardScaler().fit_transform(features)

xtrain, xtest, ytrain, ytest = train_test_split(features, target, random_state=2, test_size=0.2)
def train(xtrain, ytrain, epochs):

    w1 = np.random.random((features.shape[1], 256)) - 0.5
    w2 = np.random.random((256, 128)) - 0.5
    w3 = np.random.random((128, 3)) - 0.5

    for epoch in range(epochs):
        errors = 0
        for i in range(len(xtrain)):
            ytarget = ytrain[i: i+1]

            inpt = xtrain[i: i+1]
            layer1 = sigmoid(np.dot(inpt, w1))
            layer2 = sigmoid(np.dot(layer1, w2))
            outpt = sigmoid(np.dot(layer2, w3))
            #print(outpt, end='----------\n')

            error = np.sum(loss_func(outpt, ytarget))
            errors += error

            # back prop
            delta3 = (ytarget - outpt) * dersigmoid(outpt)
            delta2 = delta3.dot(w3.T) * dersigmoid(layer2)
            delta1 = delta2.dot(w2.T) * dersigmoid(layer1)

            w3 += layer2.T.dot(delta3)
            w2 += layer1.T.dot(delta2)
            w1 += inpt.T.dot(delta1)
        
    return w1, w2, w3

w1, w2, w3 = train(xtrain, ytrain, 250)
def predict(xtest, w1, w2, w3):
    layer1 = sigmoid(np.dot(xtest, w1))
    layer2 = sigmoid(np.dot(layer1, w2))
    outpt = sigmoid(np.dot(layer2, w3))
    return np.argmax(outpt)
def accuracy(ytest, ypred):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == ypred[i]:
            correct += 1
    return correct / len(ytest)
results = []

for i in range(len(xtest)):
    prediction = predict(xtest[i: i+1], w1, w2, w3)
    results.append(prediction)
y_tests = []
ytest = list(ytest)
for i in range(len(ytest)):
    y_tests.append(np.argmax(ytest[i]))   
accuracy(y_tests, results)
results = []

for i in range(len(xtrain)):
    prediction = predict(xtrain[i: i+1], w1, w2, w3)
    results.append(prediction)

y_tests = []
ytest = list(ytrain)
for i in range(len(ytrain)):
    y_tests.append(np.argmax(ytrain[i]))  

accuracy(y_tests, results) 
def confusion_matrix(ytest, ypred):
    matrix = np.zeros((3,3))
    for i in range(len(ytest)):
        if ytest[i] == ypred[i]:
            if ytest[i] == 0:
                matrix[0][0] += 1
            else:
                matrix[1][1] += 1
        else:
            if ytest[i] == 0:
                matrix[0][1] += 1
            else:
                matrix[1][0] += 1
    return matrix


confusion_matrix(y_tests, results)
