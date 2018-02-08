# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/2/6$ 20:52$
# @Author  : Richard Yang
# @Email   : 971914443@qq.com
# @File    : regTrees$.py
# Description :
# --------------------------------------

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

np.set_printoptions(suppress=True)#这一句主要是把科学计数法转换为浮点数

def binSplitDataSet(dataSet,feature,value):
    #!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : dataSet:数据集
    # feature:待切分的特征
    # value:该特征的值 如果大于特定的值，那么就生成右子树，反之，生成左子树.
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :切分数据集.
    # --------------------------------------
    mat0=dataSet[np.nonzero(dataSet[:,feature] > value)[0],:]
    mat1 = dataSet[np.nonzero(dataSet[:, feature] <=value)[0], :]
    return mat0,mat1
def loadDataSet(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : filename：文件路径
    # @Returns  : dataMat：数据矩阵
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    # f=open(fileName,'r')
    # dataMat=[]
    # for line in f.readlines():
    #     line=line.strip().split('\t')#先去掉空格，完了以后 将\t为分隔符,将数据读进来.
    #     line=list(map(float,line))
    #     dataMat.append(line)
    dataMat=np.loadtxt(fileName,dtype=float,delimiter='\t')#返回的矩阵
    return dataMat
def plotDataSet(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : dataMat：矩阵
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    #提前说一下，我这么写 就是使用两种方法实现了 机器学习实战上的返回的list类型，我写的这个返回的矩阵类型
    #都可以，但是我这个效率高一点，直接进行矩阵的切分，而列表只能进行for循环.
    dataMat=loadDataSet(fileName)
    #print(dataMat.shape) (200,2)
    x_list=list(dataMat[:,0])
    y_list=list(dataMat[:,1])
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(1,1,1)#写成 ax=fig.add_subplot(111)
    ax.scatter(x_list,y_list,s=20,c='b',marker='o',alpha=0.5)#alpha指的是透明度
    plt.title('DataSet')
    plt.xlabel('X')
    plt.subplots_adjust()
    plt.show()
def plotDataSet1(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : dataMat：矩阵
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    #提前说一下，我这么写 就是使用两种方法实现了 机器学习实战上的返回的list类型，我写的这个返回的矩阵类型
    #都可以，但是我这个效率高一点，直接进行矩阵的切分，而列表只能进行for循环.
    dataMat=loadDataSet(fileName)
    #print(dataMat.shape) (200,2)
    x_list=list(dataMat[:,1])
    y_list=list(dataMat[:,2])
    fig=plt.figure(figsize=(8,6))
    ax=fig.add_subplot(1,1,1)#写成 ax=fig.add_subplot(111)
    ax.scatter(x_list,y_list,s=20,c='b',marker='o',alpha=0.5)#alpha指的是透明度
    plt.title('DataSet')
    plt.xlabel('X')
    plt.subplots_adjust()
    plt.show()
def regLeaf(dataSet):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : dataSet :待分的数据集
    # @Returns  : Cm 表示的连续变量的平均值.
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    return np.mean(dataSet[:,-1])#cm为目标变量的均值 因为是处理的连续的变量.
def regErr(dataSet):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    return np.var(dataSet[:,-1])*np.shape(dataSet)[0]
def createTree(dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : dataSet：数据集
    # @Returns  : 回归树
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    feat,val=chooseBestSplit(dataSet,leafType,errType,ops)
    if feat==None:
        return val
    reTree={}
    reTree['spInd']=feat
    reTree['spVal']=val
    lSet,rSet=binSplitDataSet(dataSet,feat,val)
    # 你可能会疑惑 之前的那一行还存在，特征是不是没有进行删除，你把数据集分开，你会发现在之前的那一行的数据 类型都报纸了一致
    #所以没必要把用过的特征的那一列删掉，而是保证分开的数据集之前分开的特征保持一致就可以了.
    reTree['left']=createTree(lSet,leafType,errType,ops=(1,4))
    reTree['right']=createTree(rSet,leafType,errType,ops=(1,4))
    return reTree
def chooseBestSplit(dataSet, leafType=regLeaf, errType=regErr, ops=(1, 4)):
    #     #ops设定了tolS和tolN tolS

    # tolS允许的误差下降值,tolN切分的最少样本数
    tolS = ops[0];
    tolN = ops[1]
    # 如果当前所有值相等,则退出。(根据set的特性)
    if len(set(dataSet[:, -1].T.tolist()[0])) == 1:
        return None, leafType(dataSet)
    # 统计数据集合的行m和列n    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    m, n = np.shape(dataSet)
    # 默认最后一个特征为最佳切分特征,计算其误差估计
    S = errType(dataSet)
    # 分别为最佳误差,最佳特征切分的索引值,最佳特征值
    bestS = float('inf');
    bestIndex = 0;
    bestValue = 0
    # 遍历所有特征列
    for featIndex in range(n - 1):
        # 遍历所有特征值
        for splitVal in set(dataSet[:, featIndex].T.A.tolist()[0]):
            # 根据特征和特征值切分数据集
            mat0, mat1 = binSplitDataSet(dataSet, featIndex, splitVal)
            # 如果数据少于tolN,则退出
            if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN): continue
            # 计算误差估计
            newS = errType(mat0) + errType(mat1)
            # 如果误差估计更小,则更新特征索引值和特征值
            if newS < bestS:
                bestIndex = featIndex
                bestValue = splitVal
                bestS = newS
    # 如果误差减少不大则退出
    if (S - bestS) < tolS:
        return None, leafType(dataSet)
    # 根据最佳的切分特征和特征值切分数据集合
    mat0, mat1 = binSplitDataSet(dataSet, bestIndex, bestValue)
    # 如果切分出的数据集很小则退出
    if (np.shape(mat0)[0] < tolN) or (np.shape(mat1)[0] < tolN):
        return None, leafType(dataSet)
    # 返回最佳切分特征和特征值
    return bestIndex, bestValue
def isTree(obj):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : obj:输入tree的对象。
    # @Returns  : boolean类型的对象
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    return (type(obj).__name__=='dict')
def getMean(tree):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : tree
    # @Returns  : 平均值
    # @Author   : Richard Role
    # @File    : regTrees.py
    # Description :使用了塌陷处理(既返回树平均值)
    # --------------------------------------
    if isTree(tree['right']):tree['right']=getMean(tree['right'])
    if isTree(tree['left']):tree['left']=getMean(tree['left'])
    return (tree['left']+tree['right'])/2.0
def prune(tree,testData):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : tree 决策树 testData:测试集
    # @Returns  : 树的平均值
    # @Author   : Richard Yang
    # @File    : regTrees.py
    # Description :
    # --------------------------------------
    # 如果测试集为空,则对树进行塌陷处理
    # 如果测试集为空,则对树进行塌陷处理
    if np.shape(testData)[0] == 0: return getMean(tree)
    # 如果有左子树或者右子树,则切分数据集
    if (isTree(tree['left']) and isTree(tree['right'])):
        lSet,rSet=binSplitDataSet(testData, tree['spInd'], tree['spVal'])
        # 处理左子树(剪枝)
        if isTree(tree['left']):tree['left']=prune(tree['left'],lSet)
        #处理右子树
        if isTree(tree['right']): tree['right'] = prune(tree['right'], rSet)
    #如果当前的左右节点为叶节点
    if not isTree(tree['left']) and not isTree(tree['right']):
        #计算误差
        lSet,rSet=binSplitDataSet(testData,tree['spInd'],tree['spVal'])
        #求方差 所以 是分开的数据集减去均值.
        #计算没有合并的误差
        errorNoMerge=sum(np.power(lSet[:,-1]-tree['left'],2))+\
            sum(np.power(rSet[:,-1]-tree['right'],2))
        treeMean=(tree['left']+tree['right'])/2
        #计算合并的误差
        errorMerge = np.sum(np.power(testData[:, -1] - treeMean, 2))
        if errorMerge<errorNoMerge:
            print('Mergeing')
            return treeMean
        else: return tree
    else :return tree
if __name__ == '__main__':
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False  # 正确显示正负号
    # testMat=np.mat(np.eye(4))
    # mat0,mat1=binSplitDataSet(testMat,1,0.5)
    # print('原始集合:\n',testMat)
    # print('mat0:\n',mat0)
    # print('mat1:\n',mat1)
    # fileName='ex00.txt'
    # plotDataSet(fileName)
    # myDat=loadDataSet('ex0.txt')
    # myMat=np.mat(myDat)
    # print(createTree(myMat))
    # fileName = 'ex0.txt'
    # plotDataSet1(fileName)
    # myDat = loadDataSet(fileName)
    # myMat=np.mat(myDat)
    # print(createTree(myMat))
    train_filename = 'ex2.txt'
    train_Data = loadDataSet(train_filename)
    train_Mat = np.mat(train_Data)
    tree = createTree(train_Mat)
    print(tree)
    test_filename = 'ex2test.txt'
    test_Data = loadDataSet(test_filename)
    test_Mat = np.mat(test_Data)
    print(prune(tree, test_Mat))

