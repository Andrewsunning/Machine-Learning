import numpy as np
import operator
from os import listdir

def createDataSet():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    return group, labels


def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines, 3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index, :] = listFromLine[0: 3]
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat, classLabelVector

# 归一化特征值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    row = dataSet.shape[0]
    normDataSet = dataSet - np.tile(minVals, (row, 1))
    normDataSet = normDataSet / np.tile(ranges, (row, 1))
    return normDataSet, ranges, minVals

# 模型验证，使用百分之hoRatio的训练集做测试集，其中kNN的最近邻参数是k=3
def datingClassTest(hoRatio=0.1, k=3):
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    row = normMat.shape[0]
    numTestVecs = int(row * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs: row, :], datingLabels[numTestVecs: row], k)
        print('the classifier came back with: {}, the real number is: {}'.format(classifierResult, datingLabels[i]))
        if  (classifierResult != datingLabels[i]): errorCount += 1.0
    print('the total error rate is: {}'.format(errorCount/float(numTestVecs)))


def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input('percentage of time spent playing video games?'))
    ffMiles = float(input('frequent flier miles earned per year?'))
    iceCream = float(input('liters of ice cream consumed per year?'))
    datingDataMat, datingLabels = file2matrix('./datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = np.array([ffMiles, percentTats, iceCream])
    classifierResult =classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print('You will probably like this person: {}'.format(resultList[classifierResult-1]))

def img2vector(filename):
    returnVect = np.zeros((1, 1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, i*32 + j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./digits/trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m, 1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileName = fileNameStr.split('.')[0]
        classNumStr = fileName.split('_')[0]
        hwLabels.append(classNumStr)
        trainingMat[i, :] = img2vector('./digits/trainingDigits/{}'.format(fileNameStr))
    testFileList = listdir('./digits/testDigits')
    mTest = len(testFileList)
    errorCount = 0.0
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileName = fileNameStr.split('.')[0]
        classNumStr = fileName.split('_')[0]
        vectorUnderTest = img2vector('./digits/testDigits/{}'.format(fileNameStr))
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print('the classifier came back with {}, thr real answer is {}'.format(classifierResult, classNumStr))
        if classifierResult != classNumStr : errorCount += 1.0
    print('the total number of errors is {}'.format(errorCount))
    print('the total error rate is {}'.format(errorCount/float(mTest)))




