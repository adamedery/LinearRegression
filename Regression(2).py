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

alpha = 0.02

params = (xTrain, yTrain, alpha)

# optimizing using the function defined
optRes = minCalc(my_func, weights, params,options = {'maxiter' : 10000})

if not optRes.success:
    print('Optimization failed. Continuing with final iteration of weights.')

weights = optRes.x

#calculate training, validation and test error
trainError = meanSquaredError(xTrain, yTrain, weights)
validError = meanSquaredError(xValid, yValid, weights)
testError = meanSquaredError(xTest, yTest, weights)

# printing the errors to the screen
print('Training error:', trainError)
print('Validation error:', validError)
print('Test error:', testError)

#plotting the samples from all sets
plt.plot(xTrain, yTrain, 'ro', markersize = 3)
plt.plot(xValid, yValid, 'bs', markersize = 3)
plt.plot(xTest, yTest, 'mo', markersize = 3)

#plotting the final version of the function over the dataset
x = np.arange(-1,1,0.01)
y = []
for i in range(len(x)):
    y.append(evaluation(weights, x[i]))
plt.plot(x, y, 'y-')

#plot formatting
plt.axis([-1,1,-30,30])
plt.xlabel('Input')
plt.ylabel('Target')
plt.title('Best alpha fit')
plt.show()
