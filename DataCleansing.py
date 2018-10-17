x = []
y = []
identifiers = []

with open("Datasets/communities.data", "r") as file:
    # read data
    data = file.read()

    #split data into samples
    samplesSplit = data.split('\n')

    #split smaples into features
    for i in samplesSplit:
        x.append(i.split(','))

    #delete last, empty samples
    del x[-1]

    #this sample has a missing value that none of the other samples are missing so I deleted it
    del x[130]

    #remove the target feature, and the first 4 elements of each sample, as they are simply the identifier
    for i in x:
        y.append(i[-1])
        identifiers.append(i[:4])
        del i[:4]
        del i[-1]

    #convert all values to float andcalculates the mean of the features and uses it to replace all of the smaples with missing values
    features = []
    for j in range(len(x[0])):
        for i in range(len(x)):
            if x[i][j] == '?':
                features.append(j)
                break

    for i in x:
        for j in features[::-1]:
            del i[j]

    print("features removed: " + str(features))

#saves the changes to a new file
with open("Datasets/communitiesUsableB.data", "w") as file:
    for i in range(len(x)):
        #file.write(','.join(identifiers[i]) + ',')
        file.write(','.join(map(str, x[i])))
        file.write(',' + y[i] + '\n')
