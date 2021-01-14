# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:21:45 2020
主函数
@author: Wang
"""

from ScheduleAlgirithm import *
from TaskSchedule import *

dataString="2020-04-30" # 使用数据集
waitList,cpuTime = dataPross(dataString)
nodeNum=max(300,np.max(waitList[:,6])+100)
result_fcfs=FCFS(nodeNum,waitList,cpuTime)

weight=[1,1,1,1,1,1,1] # 权重系数
myPriority(nodeNum,waitList,cpuTime,weight)
