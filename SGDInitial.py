import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt

# reads the data from the given data files
trainData = csv.reader(open("Datasets/Dataset_2_train.csv", "r"))
validData = csv.reader(open("Datasets/Dataset_2_valid.csv", "r"))
testData = csv.reader(open("Datasets/Dataset_2_test.csv", "r"))

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

alpha = 1e-4
w = [1,1]

validError = []
trainError = []

numEpochs = 100

#uses stochastic gradient descent to optimize the problem and prints the errors to the screen at every epoch
for i in range(numEpochs):
    print('EPOCH ', i)

    for j in range(len(xTrain)):
        temp = -1 * alpha * (w[0] + xTrain[j] * w[1] - yTrain[j])
        w[0] = w[0] + temp
        w[1] = w[1] + temp * xTrain[j]

    sumError = 0

    for j in range(len(xValid)):
        sumError += (w[0] + w[1] * xValid[j] - yValid[j]) ** 2
    validError.append(sumError / len(xValid))
    print('Validation error: ', sumError / len(xValid))

    sumError = 0

    for j in range(len(xTrain)):
        sumError += (w[0] + w[1] * xTrain[j] - yTrain[j]) ** 2
    trainError.append(sumError / len(xTrain))
    print('Training error: ', sumError / len(xTrain))

#plots the errors vs the epoch
plt.plot(list(range(numEpochs)), validError, 'r-', markersize = 1)
plt.plot(list(range(numEpochs)), trainError, 'b-', markersize = 1)

# plot formatting
plt.axis([0, numEpochs,0,40])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Epoch')
plt.show()
