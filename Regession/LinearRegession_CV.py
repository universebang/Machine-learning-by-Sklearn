# -*- coding: utf-8 -*-
# --------------------------------------
# @Time    : 2018/2/15$ 15:57$
# @Author  : Richard Yang
# @Email   : 971914443@qq.com
# @File    : LinearRegession_CV$.py
# Description :
# --------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, Ridge
from sklearn.model_selection import GridSearchCV

def linearRegession_analysis(fileName):
    ##!/usr/bin/python3.6
    # -*- coding: utf-8 -*-
    # --------------------------------------
    # @Parameters   : fileName:文件路径名
    # @Returns  : jhj
    # @Author   : Richard Yang
    # @File    : LinearRegession_CV.py
    # Description :线性回归
    # --------------------------------------
    #panads读入
    data=pd.read_csv(fileName)
    x=data[['TV','Radio','Newspaper']]
    y=data['Sales']

    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
    #岭回归是带二范数惩罚的线性回归
    #lasso回归则是带一范数的线性回归
    model=Lasso()

    alpha_can = np.logspace(-3, 2, 10)
    lasso_model = GridSearchCV(model, param_grid={'alpha': alpha_can}, cv=5)
    lasso_model.fit(x, y)
    print('验证参数：\n', lasso_model.best_params_)

    y_hat=lasso_model.predict(np.array(x_test))
    mse=np.average((y_hat-np.array(y_test))**2)
    rmse=np.sqrt(mse)
    print('均方根误差:',rmse)
    print('均方误差:',mse)

    t=np.arange(len(x_test))
    plt.plot(t,y_test,'r-',linewidth=2,label='Test')
    plt.plot(t,y_hat,'g-',linewidth=2,label='predict')
    plt.legend(loc='upper left')
    plt.grid()
    plt.show()
if __name__ == '__main__':
    fileName='8.Advertising.csv'
    linearRegession_analysis(fileName)

