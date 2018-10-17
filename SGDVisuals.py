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

numEpochs = 100

#uses stochastic gradient descent to optimize the problem using the given step size, plots the fit every 10 epochs, and prints the test MSE at the end
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

        if i % 10 == 0:
            plt.figure(i//10)
            x = np.arange(0,2,0.1)
            y = []
            for j in range(len(x)):
                y.append(w[0] + x[j] * w[1])
            plt.plot(x, y, 'y-')
            plt.plot(xTrain, yTrain, 'ro', markersize = 2)
            plt.plot(xValid, yValid, 'bs', markersize = 2)
            plt.axis([0, 2, 0, 15])
            plt.xlabel('Input')
            plt.ylabel('Target')
            plt.title('Fit for epoch ' + str(i))
            plt.show()

    testSum = 0
    for j in range(len(xTest)):
        testSum += (w[0] + w[1] * xTest[j] - yTest[j]) ** 2
    print(testSum / len(xTest))

    return results

validError = MSECompute(alpha)
