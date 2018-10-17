import csv
import numpy as np
from scipy.optimize import minimize as minCalc
import matplotlib.pyplot as plt

# reads the data from the given data files
trainData = csv.reader(open("Datasets/Dataset_1_train.csv", "r"))
validData = csv.reader(open("Datasets/Dataset_1_valid.csv", "r"))
testData = csv.reader(open("Datasets/Dataset_1_test.csv", "r"))

# these 3 repeated sections convert the data to float values and append it to the input and target arrays of the right set
xTrain = []
yTrain = []
for row in trainData:
    xTrain.append(float(row[0]))
    yTrain.append(float(row[1]))

xValid = []
yValid = []
for row in validData:
    xValid.append(float(row[0]))
    yValid.append(float(row[1]))

xTest = []
yTest = []
for row in testData:
    xTest.append(float(row[0]))
    yTest.append(float(row[1]))

# method for evaluating the prediction of your model given the weights and an input
def evaluation(weights, x):
    result = 0
    for i in range(len(weights)):
        result += weights[i] * (x ** i)
    return result

# method for finding the mean squared error of the function given the input, output, weights and a regularization coefficent using L2 regularization and the evaluation function
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

# method for finding the MSE of the output given input, target and weights arrays
def meanSquaredError(inputList, outputList, weights):
    sumError = 0
    for i in range(len(inputList)):
        sumError += (evaluation(weights, inputList[i]) - outputList[i]) ** 2
    sumError = sumError / len(inputList)
    return sumError

weights = [1 for i in range(21)]

alphas = [0.01,0.015,0.02,0.025,0.03,0.04,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,1]

trainErrors = []
validErrors = []

# finds the optimal solution for minimizing the Loss function using ridge regression, and stores the training and validation errors
for alpha in alphas:
    weights = [1 for i in range(21)]

    params = (xTrain, yTrain, alpha)

    optRes = minCalc(my_func, weights, params)

    if not optRes.success:
        print('Optimization failed. Continuing with final iteration of weights.')

    weights = optRes.x

    trainError = meanSquaredError(xTrain, yTrain, weights)
    validError = meanSquaredError(xValid, yValid, weights)

    print('Training error:', trainError)
    print('Validation error:', validError)

    trainErrors.append(trainError)
    validErrors.append(validError)


# ploting the error in the training and validation sets vs values of alpha chosen
plt.plot(alphas, trainErrors, 'ro', markersize = 3)
plt.plot(alphas, validErrors, 'bs', markersize = 3)

#plot formatting
plt.axis([0,1,0,25])
plt.xlabel('Alpha')
plt.ylabel('MSE')
plt.title('Regularization error')
plt.show()
