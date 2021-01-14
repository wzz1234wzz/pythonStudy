# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:34:27 2020

@author: Wang
"""


# -*- coding: utf-8 -*-
"""kkCreated on Thu Apr 23 18:24:16 2020
实时调度
@author: Wang
"""
import numpy as np
from TaskSchedule import *

nodeNum=10     # 节点的总数目
control=1      # 1：作图止于当前时刻 2：作图至于完成时刻 3：1和2综合
currentTime=0  # 当前时刻, 以第一个作业提交时刻为初始0点
weight=[1,1,1,1,1,1,1] # 权重系数

# 0作业ID 1用户等级 2用户提交次数 3已使用时长 4作业提交时刻 
# 5被抢占个数 6所需节点 7等待时长 8所属用户
waitList=np.array([(0,1,1,0,0,0,2,0,1),
                   (1,1,1,0,0,0,3,0,1),
                   (2,1,2,0,5,0,3,0,1),
                   (3,2,1,0,5,0,2,0,2),
                   (4,2,1,0,5,0,4,0,2),
                   (5,1,1,0,10,0,2,0,3),
                   (6,3,1,0,10,0,3,0,3),
                   (7,2,1,0,20,0,3,0,3)])
waitNum=waitList.shape[0]      # 等待列表作业数目

# 计算当前阶段的优先级
userType=np.unique(waitList[:,8]) # 用户类型
cpuTime=np.array([10,20,300,20,10,150,30,50]) # cup使用时间

TimeTable=np.zeros((nodeNum,200),dtype=int) # 用于记录占用情况
TimeTable2=np.zeros((nodeNum,200),dtype=int)
result=[]
maxFinish=0
while waitNum:
    accessNodeNum=len(np.where(TimeTable[:,currentTime]==0)[0]) # 找出空闲节点
    if len(np.where(waitList[:,6]<=accessNodeNum)[0])>0:
        priority=currentPriority(waitList,currentTime,accessNodeNum,weight,waitNum)
        sortIndex=np.argsort(-priority) # 获取降序索引
        waitList=waitList[sortIndex[0],:]
        cpuTime=cpuTime[sortIndex[0]]
        # 判断当前时刻是否存在最优任务可用的节点
        while waitList[0,4]<=currentTime: # 所调度的作业必须已
            randFinishTime=currentTime+cpuTime[0]
            maxFinish=max([maxFinish,randFinishTime])
            fassibleIndex=np.where(TimeTable[:,currentTime]==0)[0] # 找出空闲节点
            if len(fassibleIndex)>=waitList[0,6]:                  # 如果可用节点数目满足作业所需节点
                TimeTable[fassibleIndex[0:waitList[0,6]],currentTime:randFinishTime]=1
                TimeTable2[fassibleIndex[0:waitList[0,6]],currentTime:randFinishTime]=100+waitList[0,0]
                temp_result=list(waitList[0,:])                    # 0-8
                temp_result.append(currentTime)                    # 9实际开始时间
                temp_result.append(randFinishTime)              # 10实际结束时间
                temp_result.append(currentTime-waitList[0,4])      # 11延迟时间
                temp_result.append(fassibleIndex[0:waitList[0,6]]) # 12所分配的节点
                result.append(temp_result)
                
                # 更新被抢占次数
                stopIndex=1+np.where(waitList[1:,4]<=waitList[0,4])[0]
                waitList[stopIndex,5]+=1 # 时间提的早却没分上
                
                waitList=np.copy(waitList[1:,:]) # 将已开始执行的任务删除
                cpuTime=np.copy(cpuTime[1:]) 
                
                waitNum-=1 # 更新等待作业队列数目
                if waitNum==0:
                    break
            else: # 时间继续进行
                break # 没有满足数目的节点
    #plotTake(result,currentTime,nodeNum,control) # 作当前时刻调度甘特图
    
    # 更新列表中的其他作业信息
    finishTime=0 # 计算至当前时刻，用户所有所用的时间
    for i in userType:
        for value in result:
            if value[8]==i: # 同一用户
                finishTime+=currentTime-value[9]
        sameUser=1+np.where(waitList[1:,8]==i)[0]
        waitList[sameUser,3]=finishTime
    currentTime+=1
    waitList[:,7]=currentTime-waitList[:,4] # 更新等待时长
    errIndex=np.where(waitList[:,7]<0)
    waitList[errIndex,7]=0
    print("当前时刻：",currentTime)

while currentTime<=maxFinish:
    #plotTake(result,currentTime,nodeNum,control)
    currentTime+=1
    print("当前时刻：",currentTime)
plotTake(result,currentTime,nodeNum,control)

evaluate(result) # 计算评价指标

