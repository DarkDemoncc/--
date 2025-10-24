python
运行
from numpy import *
import matplotlib.pyplot as plt
import operator

def createDataSet():
    group = array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])  # 修正原数组维度不一致问题
    labels = ["A", "A", "B", "B"]
    return group, labels

def file2matrix(filename):
    fr = open(filename)
    array_olines = fr.readlines()
    number_lines = len(array_olines)
    return_mat = zeros((number_lines, 3))
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
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet / tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def classify0(inX, dataSet, labels, k):
    # KNN分类核心算法
    dataSetSize = dataSet.shape[0]
    # 计算欧氏距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 按距离排序
    sortedDistIndicies = distances.argsort()
    # 选择距离最近的k个点
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序并返回出现次数最多的类别
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def datingClassTest():
    hoRatio = 0.50  # 测试集比例
    datingDataMat, datingLabels = file2matrix(r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :], 
                                    datingLabels[numTestVecs:m], 3)
        print(f"预测结果: {classifierResult}, 实际结果: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    print(f"错误率: {errorCount / float(numTestVecs):.2%}")

def classify_person():
    resultList = ['不喜欢', '一般喜欢', '非常喜欢']
    # 交互式输入特征
    percentTats = float(input("玩视频游戏所耗时间百分比: "))
    ffMiles = float(input("每年获得的飞行常客里程数: "))
    iceCream = float(input("每周消费的冰淇淋公升数: "))
    
    # 加载数据并归一化
    datingDataMat, datingLabels = file2matrix(r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt")
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # 构造输入向量并归一化
    inArr = array([ffMiles, percentTats, iceCream])
    normInArr = (inArr - minVals) / ranges
    
    # 进行分类预测
    classifierResult = classify0(normInArr, normMat, datingLabels, 3)
    print(f"你对这个人的印象可能是: {resultList[classifierResult - 1]}")

if __name__ == "__main__":
    # 显示特征散点图
    file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
    dating_data_mat, dating_labels = file2matrix(file_path)
    
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 
               15.0 * array(dating_labels), 15.0 * array(dating_labels))
    ax.set_xlabel('玩视频游戏所耗时间百分比')
    ax.set_ylabel('每周消费的冰淇淋公升数')
    ax.set_title('特征关系散点图')
    plt.show()
    
    # 执行分类测试
    print("\n===== 分类测试结果 =====")
    datingClassTest()
    
    # 死循环提供交互式预测
    print("\n===== 开始交互式预测 (输入q退出) =====")
    while True:
        user_input = input("\n是否进行新的预测? (y/n): ")
        if user_input.lower() == 'n' or user_input.lower() == 'q':
            print("程序结束，再见!")
            break
        elif user_input.lower() == 'y':
            classify_person()
        else:
            print("请输入y或n")
