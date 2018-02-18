# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/2/13$ 10:17$
# @Author  : Richard Role
# @Email   : 971914443@qq.com
# @File    : Advertising$.py
# Description :
# --------------------------------------

import numpy as np
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings
def input_data_csv(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName
    # @Returns  : no
    # @Author   : Richard Yang
    # @File    : Advertising.py
    # Description :读取数据
    # --------------------------------------
      # 顾名思义 csv就是comma seperate value 就是使用逗号隔开形式的文件
    # python自带读取csv文件的读取方法
    f = open(fileName)
    x = []
    y = []
    for i, d in enumerate(f):
        if i == 0:
            continue
        d = d.strip()  # 去掉空格
        if not d:
            continue  # 某一行为空
        d = list(map(float,d.split(',')))
        print(d)
    #     x.append(d[1:-1])
    #     y.append(d[-1])
    # print(x)
    # print(y)


def input_data_open(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName 文件的路径名
    # @Returns  : no
    # @Author   : Richard Yang
    # @File    : Advertising.py
    # Description :python 自带库
    # --------------------------------------
    x=[]
    y=[]
    f = open(fileName, 'r')
    f = csv.reader(f)
    for i,d in enumerate(f):
        # print(d)
        if i==0:
            continue
        if not d:
            continue
        d=[float(e) for e in d]
        x.append(d[1:-1])
        y.append(d[-1])
    print(x)
    print(y)
def input_data_numpy(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName:文件路径
    # @Returns  : no
    # @Author   : Richard Yang
    # @File    : Advertising.py
    # Description :使用numpy来读写数据
    # --------------------------------------
    x=[]
    y=[]
    p = np.loadtxt(fileName, dtype=np.float32,delimiter=',', skiprows=1)
    for line in p:
        x.append(list(line[1:-1]))
        y.append(line[-1])
    print(x)
    print(y)
def input_data_pandas(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName
    # @Returns  : no
    # @Author   : Richard Role
    # @File    : Advertising.py
    # Description :
    # --------------------------------------
    data=pd.read_csv(fileName)
    x = data[['TV', 'Radio', 'Newspaper']]
    #x=data[['TV','Radio']]
    y=data['Sales']
    return x,y
def data_visualization():
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : no
    # @Returns  : no
    # @Author   : Richard Yang
    # @File    : Advertising.py
    # Description :数据可视化
    # --------------------------------------
    fileName = '8.Advertising.csv'
    # input_data_open(fileName)
    # input_data_numpy(fileName)
    # print('================================')
    # input_data_csv(fileName)

    data_x,data_y=input_data_pandas(fileName)
    #绘制一
    # plt.plot(data_x['TV'],data_y,'ro',label='TV')
    # plt.plot(data_x['Radio'],data_y,'g^',label='Radio')
    # plt.plot(data_x['Newspaper'],data_y,'mv',label='Newspaper')
    # plt.legend(loc='lower right')
    # plt.grid()
    # plt.show()

    #绘制二
    fig=plt.figure()
    ax1=fig.add_subplot(221)
    ax1.plot(data_x['TV'],data_y,'ro',label='TV')
    plt.legend(loc='lower right')
    ax2=fig.add_subplot(222)
    ax2.plot(data_x['Radio'],data_y,'g^',label='Radio')
    plt.legend(loc='lower right')
    ax3=fig.add_subplot(223)
    ax3.plot(data_x['Newspaper'],data_y,'mv',label='Newspaper')
    plt.legend(loc='lower right')
    plt.suptitle('不同特征的数据可视化')
    plt.subplots_adjust()
    plt.grid()
    #plt.tight_layout()
    plt.show()

    # 绘制3
    # plt.figure(figsize=(9,12))
    # plt.subplot(311)
    # plt.plot(data_x['TV'], data_y, 'ro')
    # plt.title('TV')
    # plt.grid()
    # plt.subplot(312)
    # plt.plot(data_x['Radio'], data_y, 'g^')
    # plt.title('Radio')
    # plt.grid()
    # plt.subplot(313)
    # plt.plot(data_x['Newspaper'], data_y, 'b*')
    # plt.title('Newspaper')
    # plt.grid()
    # plt.tight_layout()
    # plt.show()

    x_train,x_test,y_train,y_test=train_test_split(data_x,data_y,random_state=1,train_size=0.7)#默认为0.75
    linreg = LinearRegression()
    model=linreg.fit(x_train,y_train)
    print(model)
    print(linreg.coef_)
    print(linreg.intercept_)

    y_hat=model.predict(np.array(x_test))
    mse = np.average((y_hat - np.array(y_test)) ** 2)## Mean Squared Error
    rmse=np.sqrt(mse)#Root Mean Squared Error
    print(mse,rmse)

    t = np.arange(len(x_test))
    plt.plot(t, y_test, 'r-', linewidth=2, label='Test')
    plt.plot(t, y_hat, 'g-', linewidth=2, label='Predict')
    plt.legend(loc='upper right')
    plt.grid()
    plt.show()
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    #建议画图的时候，就像模板一样，直接复制粘贴(尤其是前两行代码)
    mpl.rcParams['font.sans-serif'] = [u'SimHei']  # 黑体 FangSong/KaiTi
    mpl.rcParams['axes.unicode_minus'] = False  # 正确显示正负号
    plt.rcParams['figure.figsize'] = (8.0, 4.0)  # 设置figure_size尺寸
    plt.rcParams['image.interpolation'] = 'nearest'  # 设置 interpolation style
    plt.rcParams['image.cmap'] = 'gray'  # 设置 颜色 style
    data_visualization()



