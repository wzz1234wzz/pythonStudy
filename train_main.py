# -*- coding: utf-8 -*-
"""
Created on Mon May 11 17:55:07 2020

@author: Wang
"""

import numpy as np
import TaskSchedule as TS
import matplotlib.pyplot as plt
import PSO_train as PSO
w = 1
lr = (0.49445,1.49445)
maxgen = 100
sizepop = 20
rangepop = (0,1)
rangespeed = (-0.5,0.5)

dataString="2020-04-30" # 使用数据集
waitList,allCpuTime = TS.dataPross(dataString)
nodeNum=max(300,np.max(waitList[:,6])+100)

result=PSO.PSO_main(w,lr,maxgen,sizepop,rangepop,rangespeed,waitList,allCpuTime,nodeNum)

plt.plot(result[:,7])
plt.show()




