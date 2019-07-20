import numpy as np
import pandas as pd
'''
ref: 
1. 机器学习实战
2. 《统计学习方法》
'''

# 对数据集进行二分
def binSplitDataSet(dataSet, feature, value):
    df0 = dataSet[(dataSet.loc[:, feature] > value)]
    df1 = dataSet[(dataSet.loc[:, feature] <= value)]
    return df0, df1


# 生成叶节点
def classifyLeaf(dataSet):  # returns the value used for each leaf
    # 返回众数作为叶节点的值
    return pd.Series(dataSet.values[:, -1]).mode()[0]


# 计算划分特征选择标准，基尼系数
def calGini(dataSet):
    dataSet = dataSet.values
    y_labels = np.unique(dataSet[:, -1].T.tolist())
    y_counts = len(dataSet)
    y_prob = {}
    # 初始化基尼指数
    gini = 1.0

    # 计算每一类别的概率
    for y_label in y_labels:
        y_prob[y_label] = len(dataSet[dataSet[:, -1] == y_label]) / y_counts
        gini -= y_prob[y_label] ** 2
    return gini


# 根据基尼系数选择划分标准和划分特征值
def chooseBestSplit(dataSet, leafType=classifyLeaf, errType=calGini, ops=(0, 1)):
    tolS = ops[0];
    tolN = ops[1]
    # 如果结果为同一类，返回叶节点
    if len(set(dataSet.values[:, -1])) == 1:  # exit cond 1
        return None, leafType(dataSet)

    m, n = dataSet.shape
    # the choice of the best feature is driven by Reduction in RSS error from mean
    S = errType(dataSet)
    bestS = np.inf;
    bestIndex = 0;
    bestValue = 0
    for featIndex in range(n - 1):
        for splitVal in set(dataSet.values[:, featIndex]):
            df0, df1 = binSplitDataSet(dataSet, featIndex, splitVal)
            if (df0.shape[0] < tolN) or (df1.shape[0] < tolN): continue
            newS = errType(df0) + errType(df1)
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # if the decrease (S-bestS) is less than a threshold don't do the split
    if (S - bestS) < tolS:
        return None, leafType(dataSet)  # exit cond 2
    df0, df1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    if (df0.shape[0] < tolN) or (df1.shape[0] < tolN):  # exit cond 3
        return None, leafType(dataSet)
    return bestIndex, bestValue


# 创建CART决策树
def createTree(dataSet, leafType=classifyLeaf, errType=calGini,
               ops=(0, 1)):  # assume dataSet is NumPy Mat so we can array filtering
    feat, val = chooseBestSplit(dataSet, leafType, errType, ops)  # choose the best split
    if feat == None: return val  # if the splitting hit a stop condition return val
    retTree = {}
    retTree['spInd'] = feat
    retTree['spVal'] = val
    lSet, rSet = binSplitDataSet(dataSet, feat, val)
    retTree['left'] = createTree(lSet, leafType, errType, ops)
    retTree['right'] = createTree(rSet, leafType, errType, ops)
    return retTree


# 判断是否为子树
def isTree(obj):
    return (type(obj).__name__ == 'dict')


# 没什么用，如果测试集为空才返回该值
def getMean(tree):
    if isTree(tree['right']): tree['right'] = getMean(tree['right'])
    if isTree(tree['left']): tree['left'] = getMean(tree['left'])
    return (tree['left'] + tree['right']) / 2.0


# 对决策树进行减枝
def prune(tree, testData):
    if testData.shape[0] == 0: return getMean(tree)  # if we have no test data collapse the tree
    if (isTree(tree['right']) or isTree(tree['left'])):  # if the branches are not trees try to prune them
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])
    if isTree(tree['left']): tree['left'] = prune(tree['left'], lSet)
    if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    # 计算合并前后的基尼系数，判断是否剪枝
    if not isTree(tree['left']) and not isTree(tree['right']):
        lSet, rSet = binSplitDataSet(testData, tree['spInd'], tree['spVal'])

        errorNoMerge = calGini(lSet) + calGini(rSet)

        errorMerge = calGini(testData)
        if errorMerge < errorNoMerge:
            print("merging")
            return classifyLeaf(testData)
        else:
            return tree
    else:
        return tree

# 构建错误率计算函数
def accuracy(y_pred, y_label):
    rows = len(y_label)
    error = 0
    for i in range(rows):
        print('the classifier came back with: {}, the real number is: {}'.format(y_pred[i], y_label[i]))
        if y_label[i] != y_pred[i]:
            error += 1
    print('the total error rate is: {}'.format(error/float(rows)))

def classifyTreeEval(model, inDat):
    return float(model)


def treeForeCast(tree, inData, modelEval=classifyTreeEval):
    if not isTree(tree): return modelEval(tree, inData)

    if inData[tree['spInd']] > tree['spVal']:
        if isTree(tree['left']):
            return treeForeCast(tree['left'], inData, modelEval)
        else:
            return modelEval(tree['left'], inData)
    else:
        if isTree(tree['right']):
            return treeForeCast(tree['right'], inData, modelEval)
        else:
            return modelEval(tree['right'], inData)


def createForeCast(tree, testData, modelEval=classifyTreeEval):
    m = len(testData)
    yHat = np.zeros((m, 1))
    for i in range(m):
        yHat[i, 0] = treeForeCast(tree, testData.loc[i], modelEval)
    return yHat

if __name__=='__main__':
	# 读入数据集
	df = pd.read_table('./datingTestSet2.txt', header=None, sep='\t')
	df.head()


	# 划分训练集、验证集和测试集
	train_df = df.loc[:599, :]
	val_df = df.loc[600: 799,:].reset_index(drop=True)
	test_df = df.loc[800: ,:].reset_index(drop=True)


	# 构建决策树
	mytree = createTree(train_df)
	# 决策树剪枝
	mytree = prune(mytree, val_df)


	# 对测试集预测
	y_pred = createForeCast(mytree, test_df.loc[:, :2])
	y_pred = y_pred.reshape(-1)


	# 计算测试集错误率
	y_label = list(test_df.loc[:,3])
	y_pred = list(y_pred)
	accuracy(y_pred, y_label)

