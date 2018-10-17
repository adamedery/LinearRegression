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

alphas = [1e-6,1e-5,1e-4,1e-3,1e-2]

numEpochs = 100

#uses stochastic gradient descent to optimize the problem using the given step size and prints the test MSE to the screen at the end
def MSECompute(alpha):
    results = []
    w = [1,1]
    for i in range(numEpochs):
        for j in range(len(xTrain)):
            temp = -1 * alpha * (w[0] + xTrain[j] * w[1] - yTrain[j])
            w[0] = w[0] + temp
            w[1] = w[1] + temp * xTrain[j]

        sumError = 0

        for j in range(len(xValid)):
            sumError += (w[0] + w[1] * xValid[j] - yValid[j]) ** 2
        results.append(sumError / len(xValid))

    testSum = 0
    for j in range(len(xTest)):
        testSum += (w[0] + w[1] * xTest[j] - yTest[j]) ** 2
    print(testSum / len(xTest))

    return results

validError = []

#calculates the validation error at every step size
for i in alphas:
    validError.append(MSECompute(i))

#plots the 5 different errors of the different step sizes over the number of epochs using different markers
plt.plot(list(range(numEpochs)), validError[0], 'r-', markersize = 1)
plt.plot(list(range(numEpochs)), validError[1], 'b-', markersize = 1)
plt.plot(list(range(numEpochs)), validError[2], 'g-', markersize = 1)
plt.plot(list(range(numEpochs)), validError[3], 'y-', markersize = 1)
plt.plot(list(range(numEpochs)), validError[4], 'm-', markersize = 1)

#plot formatting
plt.axis([min(xTrain), max(xTrain), 0, 40])
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.title('MSE vs Epoch')
plt.show()
