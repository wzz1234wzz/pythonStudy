# -*- coding: utf-8 -*-
"""
    Created on Thu Apr 23 10:34:15 2020
    对整体进行调度，已知每个作业的运行时长
    @author: Wang
"""
import numpy as np

def currentPriority(waitList,currentTime,accessNodeNum,weight,waitNum):
    # 找出当前时刻及其之前的作业
    priority=np.zeros((1,waitNum),dtype=float)
    index=np.where(waitList[:,4]<=currentTime)[0]
    if len(index)>0:
        compareList=waitList[index,:]
        temp=weight[0]*compareList[:,1]/(1+sum(compareList[:,1]))+\
                 weight[1]*(1-compareList[:,2]/max(compareList[:,2]))+\
                 weight[2]*compareList[:,3]/(1+sum(compareList[:,3]))+\
                 weight[3]*(1-compareList[:,4]/(1+max(compareList[:,4])))+\
                 weight[4]*compareList[:,5]/(1+sum(compareList[:,5]))+\
                 weight[5]*(1-(abs(compareList[:,6]-accessNodeNum)/max(abs(compareList[:,6]-accessNodeNum))))+\
                 weight[6]*compareList[:,7]/(1+sum(compareList[:,7]))   
        priority[0,index]=temp
    return priority

def plotTake(result):
    import matplotlib.pyplot as plt
    import numpy as np
    color=['r','g','b','m']
    fig = plt.figure()
    for i in range(len(result)):
        for j in result[i][12]:
            plt.plot(np.array([result[i][9],result[i][10]]),np.array([j,j]),color[i%4])
            plt.plot(np.array([result[i][9],result[i][9]]),np.array([j,j+1]),color[i%4])
            plt.plot(np.array([result[i][9],result[i][10]]),np.array([j+1,j+1]),color[i%4])
            plt.plot(np.array([result[i][10],result[i][10]]),np.array([j,j+1]),color[i%4])
            plt.text(np.max([result[i][9],(result[i][10]+result[i][9])/2]),j+0.5,str(result[i][0]))
    plt.xlabel('Time')
    plt.ylabel('Node')
    plt.title("Job Scheduling")
    plt.show()
    
nodeNum=10     # 节点的总数目
currentTime=0  # 当前时刻, 以第一个作业提交时刻为初始0点

weight=[1,1,1,1,1,1,1] # 权重系数

waitList=np.array([(0,1,1,0,0,0,2,0,1),
                   (1,1,1,0,0,0,3,0,1),
                   (2,1,2,0,5,0,3,0,1),
                   (3,2,1,0,5,0,2,0,2),
                   (4,2,1,0,5,0,4,0,2),
                   (5,1,1,0,10,0,2,0,3),
                   (6,3,1,0,10,0,3,0,3),
                   (7,2,1,0,20,0,3,0,3)])
waitNum=waitList.shape[0]      # 等待列表作业数目
# 0作业ID 1用户等级 2用户提交次数 3已使用时长 4作业提交时刻 5被抢占个数 6所需节点 
# 7等待时长 8所属用户
# 计算当前阶段的优先级
userType=np.unique(waitList[:,8]) # 用户类型
# 随机产生每个任务的实际完成时间
randFinishTime=waitList[:,4]+np.random.randint(10,50,waitNum)
randFinishTime=np.array([26, 36, 41, 48, 50, 50, 60, 70])
tim=randFinishTime
TimeTable=np.zeros((nodeNum,10*max(randFinishTime)),dtype=int)
TimeTable2=np.zeros((nodeNum,10*max(randFinishTime)),dtype=int)
result=[]

while waitNum:
    accessNodeNum=len(np.where(TimeTable[:,currentTime]==0)[0]) # 找出空闲节点
    if len(np.where(waitList[:,6]<=accessNodeNum)[0])>0:
        priority=currentPriority(waitList,currentTime,accessNodeNum,weight,waitNum)
        sortIndex=np.argsort(-priority) # 获取降序索引
        waitList=waitList[sortIndex[0],:]
        randFinishTime=randFinishTime[sortIndex[0]]
        # 判断当前时刻是否存在最优任务可用的节点
        while waitList[0,4]<=currentTime: # 所调度的作业必须已
            fassibleIndex=np.where(TimeTable[:,currentTime]==0)[0] # 找出空闲节点
            if len(fassibleIndex)>=waitList[0,6]: # 如果可用节点数目满足
                TimeTable[fassibleIndex[0:waitList[0,6]],currentTime:randFinishTime[0]]=1
                TimeTable2[fassibleIndex[0:waitList[0,6]],currentTime:randFinishTime[0]]=100+waitList[0,0]
                temp_result=list(waitList[0,:])                    # 0-8
                temp_result.append(currentTime)                    # 9实际开始时间
                temp_result.append(randFinishTime[0])              # 10实际结束时间
                temp_result.append(currentTime-waitList[0,4])      # 11延迟时间
                temp_result.append(fassibleIndex[0:waitList[0,6]]) # 12所分配的节点
                result.append(temp_result)
                
                # 更新被抢占次数
                stopIndex=1+np.where(waitList[1:,4]<=waitList[0,4])[0]
                waitList[stopIndex,5]+=1 # 时间提的早却没分上
                
                waitList=np.copy(waitList[1:,:]) # 将已放置的删除
                randFinishTime=np.copy(randFinishTime[1:])
                waitNum-=1 # 更新队列数目
                if waitNum==0:
                    break
            else: # 更新各个任务的状态
                break # 没有满足数目的节点
    plotTake(result)
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
    print(currentTime)
plotTake(result)
    
    
    
    