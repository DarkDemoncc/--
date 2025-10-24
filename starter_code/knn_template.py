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


# 数据加载与可视化
file_path = r"C:\Users\Administrator\Desktop\lesson1\datingTestSet2.txt"
dating_data_mat, dating_labels = file2matrix(file_path)

plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

fig = plt.figure()
ax = fig.add_subplot(111)  
ax.scatter(dating_data_mat[:, 1], dating_data_mat[:, 2], 
           15.0*array(dating_labels), 15.0*array(dating_labels))
ax.set_xlabel('玩视频游戏所耗时间百分比')
ax.set_ylabel('每周消费的冰淇淋公升数')
ax.set_title('特征关系散点图')
# plt.show()  # 如需显示散点图可取消注释


def autoNorm(dataSet):
    # 数据归一化处理
    minVals = dataSet.min(0)  # 计算每列最小值
    maxVals = dataSet.max(0)  # 计算每列最大值
    ranges = maxVals - minVals  # 计算数据范围
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]  # 数据集行数
    normDataSet = dataSet - tile(minVals, (m, 1))  # 减去最小值
    normDataSet = normDataSet / tile(ranges, (m, 1))  # 除以范围得到归一化数据
    return normDataSet, ranges, minVals


def classify0(inX, dataSet, labels, k):
    # KNN分类核心算法
    dataSetSize = dataSet.shape[0]
    # 计算欧氏距离
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    # 距离排序（返回索引）
    sortedDistIndicies = distances.argsort()
    # 统计前k个最近邻的标签
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 排序并返回最可能的标签
    sortedClassCount = sorted(classCount.items(), 
                             key=operator.itemgetter(1), 
                             reverse=True)
    return sortedClassCount[0][0]


def datingClassTest():
    # 测试算法准确率
    hoRatio = 0.50  # 测试集比例
    datingDataMat, datingLabels = file2matrix(file_path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m * hoRatio)  # 测试集数量
    errorCount = 0.0  # 错误计数
    
    for i in range(numTestVecs):
        # 分类预测（使用后50%数据作为训练集）
        classifierResult = classify0(normMat[i, :], normMat[numTestVecs:m, :],
                                    datingLabels[numTestVecs:m], 3)
        print(f"预测结果: {classifierResult}, 实际结果: {datingLabels[i]}")
        if classifierResult != datingLabels[i]:
            errorCount += 1.0
    
    # 输出准确率
    print(f"错误率: {errorCount / float(numTestVecs):.2%}")
    print(f"准确率: {(1 - errorCount / float(numTestVecs)):.2%}")


def classify_person():
    # 交互式分类预测
    resultList = ['不喜欢', '一般喜欢', '非常喜欢']
    
    # 获取用户输入
    game_time = float(input("玩视频游戏所耗时间百分比: "))
    ice_cream = float(input("每周消费的冰淇淋公升数: "))
    fly_miles = float(input("每年获得的飞行常客里程数: "))
    
    # 加载数据并归一化
    datingDataMat, datingLabels = file2matrix(file_path)
    normMat, ranges, minVals = autoNorm(datingDataMat)
    
    # 构建输入向量并归一化
    inArr = array([fly_miles, game_time, ice_cream])
    normInArr = (inArr - minVals) / ranges  # 应用归一化
    
    # 分类预测
    classifierResult = classify0(normInArr, normMat, datingLabels, 3)
    print(f"你对这个人的印象可能是: {resultList[classifierResult - 1]}")


# 死循环调用交互式预测
if __name__ == "__main__":
    while True:
        classify_person()
        # 询问是否继续
        again = input("是否继续预测？(y/n): ")
        if again.lower() != 'y':
            print("程序结束")
            break
