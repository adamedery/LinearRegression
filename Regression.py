import csv
import numpy as np
import scipy
import matplotlib.pyplot as plt

trainData = csv.reader(open("Datasets/Dataset_1_train.csv", "r"))
validData = csv.reader(open("Datasets/Dataset_1_valid.csv", "r"))
testData = csv.reader(open("Datasets/Dataset_1_test.csv", "r"))

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

my20DPoly = np.polyfit(xTrain,yTrain,20)

def meanSquaredError(inputList, outputList, polyFunc):
    sumError = 0
    for i in range(len(inputList)):
        sumError += (np.polyval(polyFunc, inputList[i]) - outputList[i]) ** 2
    sumError = sumError / len(inputList)
    return sumError

trainError = meanSquaredError(xTrain, yTrain, my20DPoly)
validError = meanSquaredError(xValid, yValid, my20DPoly)

print(trainError)
print(validError)

plt.plot(xTrain, yTrain, 'ro')
plt.plot(xValid, yValid, 'bs')

x = np.arange(-1,1,0.01)
y = np.polyval(my20DPoly, x)
plt.plot(x, y, 'y-')
plt.axis([-1,1,-30,30])
plt.show()
