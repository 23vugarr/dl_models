import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from warnings import filterwarnings
filterwarnings('ignore')

df = pd.read_csv('lithium-ion batteries.csv')


# data cleaning
for i in range(len(df['Crystal System'])):
    if(df['Crystal System'][i] == 'triclinic'):
        df.drop(i, inplace = True, axis=0)



# data preprocessing
for i in df.columns:
    print(i, df[i].nunique())
for i in range(len(df['Has Bandstructure'])):
    if df['Has Bandstructure'][i]:
        df['Has Bandstructure'][i] = 1
    else:
        df['Has Bandstructure'][i] = 0

df['Has Bandstructure'] = df['Has Bandstructure'].astype('float64')
        
dic = { 'monoclinic': 0, 'orthorhombic': 1}

for i in range(len(df['Crystal System'])):
    df['Crystal System'][i] = dic[df['Crystal System'][i]]
    
df['Crystal System'] = df['Crystal System'].astype('float64')
df['Nsites'] = df['Nsites'].astype('float64')



#functions
def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dersigmoid(x):
    return x * (1.0 - x)

def loss_func(x, y):
    return (x - y) ** 2



# data splitting
target = df['Crystal System']
features = df.drop(columns=['Materials Id', 'Formula', 'Spacegroup', 'Crystal System'], axis=1)

target = np.array(target).reshape((-1,1))
features = np.array(features)
features = StandardScaler().fit_transform(features)

xtrain, xtest, ytrain, ytest = train_test_split(features, target, random_state=2, test_size=0.2)



# training
def train(xtrain, ytrain, epochs):

    w1 = np.random.random((features.shape[1], 256)) - 0.5
    w2 = np.random.random((256, 128)) - 0.5
    w3 = np.random.random((128, 1)) - 0.5

    for epoch in range(epochs):
        errors = 0
        for i in range(len(xtrain)):
            ytarget = ytrain[i: i+1]

            inpt = xtrain[i: i+1]
            layer1 = sigmoid(np.dot(inpt, w1))
            layer2 = sigmoid(np.dot(layer1, w2))
            outpt = sigmoid(np.dot(layer2, w3))

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

w1, w2, w3 = train(xtrain, ytrain, 53)


# testing
def predict(xtest, w1, w2, w3):
    layer1 = sigmoid(np.dot(xtest, w1))
    layer2 = sigmoid(np.dot(layer1, w2))
    outpt = sigmoid(np.dot(layer2, w3))
    return outpt


def accuracy(ytest, ypred):
    correct = 0
    for i in range(len(ytest)):
        if ytest[i] == ypred[i]:
            correct += 1
    return correct / len(ytest)
results = []


for i in range(len(xtest)):
    prediction = predict(xtest[i: i+1], w1, w2, w3)
    if(prediction > 0.5):
        prediction = 1
    else:
        prediction = 0
    results.append(prediction)

results = np.array(results).reshape((-1,1))

print(accuracy(ytest, results))

# results = []
# for i in range(len(xtrain)):
#     prediction = predict(xtrain[i: i+1], w1, w2, w3)
#     if(prediction > 0.5):
#         prediction = 1
#     else:
#         prediction = 0
#     results.append(prediction)

# results = np.array(results).reshape((-1,1))
# print(accuracy(ytrain, results))

def confusion_matrix(ytest, ypred):
    matrix = np.zeros((2,2))
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


confusion_matrix(ytest, results)
