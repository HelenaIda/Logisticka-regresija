
import numpy as np
import pandas as pd

eta=0.2

train500 = pd.read_csv("500train_BoW.csv")
train = train500.drop("TARGET",axis=1)
targets = train500["TARGET"]
test20 = pd.read_csv("20test_BoW.csv")
test = test20.drop("TARGET",axis=1)

x_input = np.array(train)

testVector = np.array(test)

noOfWeights = test.shape[1]

noOfAnimals = train.shape[0]

W = []

for n in range(noOfWeights):
      W.append(0.5)

outputs = []
errors = []

def sigma(x):
     return 1/(1+np.exp(-x))


def forwardPass(tezine, x_input, t):
     res = sigma(np.dot(tezine, x_input))
     error = 0.5 * ((t - res) ** 2)
     outputs.append(res)
     errors.append(error)
     return [res, error]

def BackPropagation(tezine,x_input,t,y_kap):
    noviW=[]
    for m in range(len(tezine)):
        curr_W=tezine[m]
        Wi=-x_input[m]*y_kap(1-y_kap)*(t-y_kap)
        noviWi=curr_W-eta*Wi
        noviW.append(noviWi)
    return noviW

def Train(trainData, weights, targets, N=2):
     W = []
     for n in range(N):
          for m in range(noOfAnimals):
               x = forwardPass(weights, trainData[m], targets[m])
               W.append(BackPropagation(weights, trainData[m],x[0],targets[m]))
     return W

def Predict(tezine,x_input):
    predikcije =[]
    for k in x_input:
        z=np.dot(tezine,x_input[k])
        res=sigma(z)
        predikcije.append(res)
    return predikcije

finalList = Train(x_input, W, targets, N = 5)[-1]
predictions = Predict(finalList, testVector)
print(predictions)

