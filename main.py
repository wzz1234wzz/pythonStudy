# -*- coding: utf-8 -*-
"""
  作业调度主函数
"""

import ScheduleAlgirithm as SA
import TaskSchedule as TS
import numpy as np
import datetime

k=3
dataString="2020-04-30" # 使用数据集
waitList,allCpuTime = TS.dataPross(dataString)
nodeNum=max(300,np.max(waitList[:,6])+100)

start = datetime.datetime.now()

if k==1:
    result_fcfs=SA.FCFS(nodeNum,waitList,allCpuTime)
elif k==2:
    threshold=30 #等待阙值，如果超过该时间，则优先级高的短作业执行
    result_fcfs_threshold=SA.FCFS_threshold(nodeNum,waitList,allCpuTime,threshold)
else:
    weight=[0.5,0.2,0.5,0.5,0.1,0.3,0.1] # 权重系数 
    weight=[1,1,1,1,1,1,1] # 权重系数
    result_myPriority=SA.myPriority(nodeNum,waitList,allCpuTime,weight)

end = datetime.datetime.now()
print (end-start)