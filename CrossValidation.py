import numpy as np
from scipy.optimize import minimize as minCalc
import matplotlib.pyplot as plt
from random import shuffle

x = []
y = []
identifiers = []

with open("Datasets/communitiesUsableB.data", "r") as file:
    # read data
    data = file.read()

    #split data into samples
    samples = data.split('\n')
    if samples[-1] == '':
        del samples[-1]

    #split smaples into features
    for sample in samples:
        x.append(sample.split(','))

    #remove the target feature, and the first 4 elements of each sample, as they are simply the identifier
    for i in x:
        y.append(float(i[-1]))
        del i[-1]
        #del i[:4]

    #convert all input values to float
    for i in range(len(x)):
        for j in range(len(x[i])):
            x[i][j] = float(x[i][j])

# removes the features we don't want to be in our array, list is in reverse order to help with indexing
badFeatures = [8,25,45,53,54,55,70,76,78,80,89,95]
for i in badFeatures[::-1]:
    for sample in x:
        del sample[i]

indexing = list(range(len(x)))

xTrain = [[],[],[],[],[]]
yTrain = [[],[],[],[],[]]
xValid = [[],[],[],[],[]]
yValid = [[],[],[],[],[]]

#randomly splits the data 5 times into different 80-20 splits
for j in range(5):
    shuffle(indexing)
    counter = 1
    for i in indexing:
        if i % 5 == 0:
            xValid[j].append(x[i])
            yValid[j].append(y[i])
            counter = 1
        else:
            xTrain[j].append(x[i])
            yTrain[j].append(y[i])
            counter += 1

#used for evaluation of the 20 degree function using give weights and inputs
def evaluation(weights, params):
    result = 0
    for i in range(len(params)):
        result += weights[i] * params[i]
    result += weights[-1]
    return result

#calculate the error of the model using ridge regression
def my_func(weights, xValues, yValues, alpha):
    error = 0
    for i in range(len(xValues)):
        error += (yValues[i] - evaluation(weights, xValues[i])) ** 2

    reg = 0
    for i in range(len(weights)):
        reg += weights[i] ** 2
    reg = alpha * reg

    result =  reg + error
    return result

#calculate the MSE of the model using given inputs and targets and weights
def meanSquaredError(inputList, outputList, weights):
    sumError = 0
    for i in range(len(inputList)):
        sumError += (evaluation(weights, inputList[i]) - outputList[i]) ** 2
    sumError = sumError / len(inputList)
    return sumError


alphas = [0.01,0.05,0.1,0.3,0.5,1]

alphas = [1]

trainErrors = []
validErrors = []
weightArray = []

for alpha in alphas:
    print('MSE\'s for alpha = ' + str(alpha) )
    for i in range(5):
        #initialize wights
        weights = [1 for j in range(len(x[0])+1)]

        params = (xTrain[i], yTrain[i], alpha)

        #optimize on the fucntion
        optRes = minCalc(my_func, weights, params, options = {'maxiter' : 10})

        if not optRes.success:
            print('Optimization failed. Continuing with final iteration of weights.')

        weights = optRes.x

        #compute train and validation errors
        trainError = meanSquaredError(xTrain[i], yTrain[i], weights)
        validError = meanSquaredError(xValid[i], yValid[i], weights)


        #print train and validation errors to screen
        print('Training error:', trainError)
        print('Validation error:', validError)

        #save train and validation errors and the weights
        trainErrors.append(trainError)
        validErrors.append(validError)
        weightArray.append(weights)

plt.figure(0)
# plots train and validation errors for the 5 splits
plt.plot(list(range(len(alphas))), [sum(trainErrors[i*5:i*5+5])/5 for i in range(len(alphas))], 'ro', markersize = 3)
plt.plot(list(range(len(alphas))), [sum(validErrors[i*5:i*5+5])/5 for i in range(len(alphas))], 'bs', markersize = 3)

# plot formatting
plt.axis([0,len(alphas),0,1000])
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('Average MSE for each alpha')

for i in range(len(alphas)):
    plt.figure(i+1)
    # plots train and validation errors for the 5 splits
    plt.plot([1,2,3,4,5], trainErrors[i*5:i*5+5], 'ro', markersize = 3)
    plt.plot([1,2,3,4,5], validErrors[i*5:i*5+5], 'bs', markersize = 3)

    # plot formatting
    plt.axis([0,5,0,1000])
    plt.xlabel('Fold number')
    plt.ylabel('MSE')
    plt.title('MSE of each fold for alpha = ' + str(alphas[i]))

plt.show()

#print the weights arrays to screen
for i in range(len(weightArray)):
    if i % 5 == 0:
        print('Weights for alpha = ' + str(alphas[int(i//5)]))
    print(weightArray[i])
