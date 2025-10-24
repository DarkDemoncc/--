from numpy import *
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ["A", "A", "B", "B"]
    return group, labels


def file2matrix(filename):
    fr = open(filename, encoding='utf-8')   # 打开文件
    array_olines = fr.readlines()           # 从文件中读取所有行
    number_lines = len(array_olines)        # 文件行数
    return_mat = zeros((number_lines, 3))   # 创建一个 number_lines×3 的矩阵
    class_label_vector = []
    index = 0
    for line in array_olines:
        line = line.strip()
        list_from_line = line.split('\t')
        return_mat[index, :] = list_from_line[0:3]
        class_label_vector.append(int(list_from_line[-1]))
        index += 1
    return return_mat, class_label_vector


def autoNorm(dataSet):
    # 数据进行归一化
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    """
    KNN分类核心函数
    inX: 输入向量（待分类样本）
    dataSet: 训练样本集
    labels: 标签
    k: 选择最近邻居的数目
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    # 使用留出法测试分类器准确率
    hoRatio = 0.50      # 测试集比例
    file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
    datingDataMat, datingLabels = file2matrix(file_path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :],
                                     normMat[numTestVecs:m, :],
                                     datingLabels[numTestVecs:m],
                                     3)
        print(f"分类结果: {classifierResult}, 真实结果: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print(f"总错误率: {errorCount/float(numTestVecs):.2%}")


def classify_person():
    """
    交互式输入三项特征，使用约会数据集做 KNN 分类，并输出结果。
    """
    resultList = ['讨厌', '有些喜欢', '非常喜欢']
    percentTats = float(input("玩视频游戏所耗时间百分比："))
    ffMiles = float(input("每年获得的飞行常客里程数："))
    iceCream = float(input("每周消费的冰淇淋公升数："))

    file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
    datingDataMat, datingLabels = file2matrix(file_path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr - minVals) / ranges, normMat, datingLabels, 3)
    print("你对这个人的感觉可能是：", resultList[classifierResult - 1])


if __name__ == "__main__":
    # 可以选择调用哪个函数
    # datingClassTest()   # 测试分类器准确率
    classify_person()     # 交互式测试
