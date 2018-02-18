# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/2/17$ 21:27$
# @Author  : Richard Yang
# @Email   : 971914443@qq.com
# @File    : Iris_LR$.py
# Description :
# --------------------------------------

import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn import preprocessing
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
def iris_type(s):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : s
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : Iris_LR.py
    # Description :对字符串的标签进行替换.
    # --------------------------------------
    it={b'Iris-setosa':0,b'Iris-versicolor':1,b'Iris-virginica':2}#为了兼容python3.x的写法
    return it[s]
def expand_axis(X1_min,X1_max,X2_min,X2_max,alpha=0.05):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : Iris_LR.py
    # Description :如果画图的时候 直接在每一列数据的最大和最小值之间画图，难免出现版面比较拥挤，所以把
    #坐标轴进行扩增。
    # --------------------------------------
    return 1.05*X1_min-0.05*X1_max,1.05*X1_max-0.05*X1_min,\
           1.05*X2_min-0.05*X2_max,1.05*X2_max-0.05*X2_min
def iris_data_visualization(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName：的路径名
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : Iris_LR.py
    # Description :会使用两种写法 去对标签进行处理，第二种方法比较少见，可以学习。
    # --------------------------------------

    data=np.loadtxt(fileName,dtype=float,delimiter=',',converters={4:iris_type})
    X, y = np.split(data, (4,), axis=1)
    print(X)
    print(y.ravel())

    #下面会使用pandas labelenco() process
    # df=pd.read_table(fileName,header=None,sep=',')
    # X,y=df.values[:,:-1],df.values[:,-1]
    # encoder=LabelEncoder()
    # y=encoder.fit_transform(y)
    # print(X)
    # print(y)

    X=X[:,:2]
    X=StandardScaler().fit_transform(X)
    lr=LogisticRegression()#分类器
    lr.fit(X,y.ravel())
    #等价于Pipline
    lr=Pipeline([('sc',StandardScaler()),
                 ('clf',LogisticRegression())])
    lr.fit(X,y.ravel())
    #画图
    N, M = 500, 500  # 横纵各采样多少个值
    X1_min,X1_max=X[:,0].min(),X[:,0].max()
    X2_min,X2_max=X[:,1].min(),X[:,1].max()
    X1_min,X1_max,X2_min,X2_max=expand_axis(X1_min,X1_max,X2_min,X2_max)
    t1=np.linspace(X1_min,X1_max,N)
    t2=np.linspace(X2_min,X2_max,M)
    x1,x2=np.meshgrid(t1,t2)
    x_test = np.stack((x1.flat, x2.flat), axis=1)  # 测试点

    cm_light = mpl.colors.ListedColormap(['#77E0A0', '#FF8080', '#A0A0FF'])
    cm_dark = mpl.colors.ListedColormap(['g', 'r', 'b'])
    y_hat=lr.predict(x_test)
    y_hat=y_hat.reshape(x1.shape)
    plt.pcolormesh(x1,x2,y_hat,cmap=cm_light)
    y=y.reshape(X[:,0].shape)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', s=50, cmap=cm_dark)  # 样本的显示
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.xlim(X1_min, X1_max)
    plt.ylim(X2_min, X2_max)
    plt.grid()
    plt.savefig('2.png')
    plt.show()

    # 训练集上的预测结果
    y_hat = lr.predict(X)
    y = y.reshape(-1)
    result = y_hat == y#返回的都是0和1  计算准确率也可以使用sum()除以总数 效果一样.
    print(y_hat)
    print(result)
    #也可以使用sklearn自带的score()
    acc = np.mean(result)
    print('准确度: %.2f%%' % (100 * acc))
if __name__ == '__main__':
    fileName='8.iris.data'
    iris_data_visualization(fileName)