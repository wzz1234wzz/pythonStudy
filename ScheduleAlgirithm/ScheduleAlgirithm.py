# -*- coding: utf-8 -*-
"""
Created on Sat May  9 14:17:07 2020

@author: Wang
"""
from TaskSchedule import *
import numpy as np

def FCFS(nodeNum,waitList,cpuTime):
    # waitList
    # 0作业ID 1用户等级 2用户提交次数 3已使用时长 4作业提交时刻 
    # 5被抢占个数 6所需节点 7等待时长 8所属用户
    currentTime=0  # 当前时刻, 以第一个作业提交时刻为初始0点
    waitNum=waitList.shape[0]      # 等待列表作业数目
    userType=np.unique(waitList[:,8]) # 用户类型
    result,maxFinish=[],0
    
    while waitNum:
        print("当前时刻：",int(currentTime)," 剩余数目：",waitNum," 已完成",len(result),"个")
        waitList=waitList[np.argsort(waitList[:,4]),:] # 排序
        cpuTime=cpuTime[np.argsort(waitList[:,4])]
        while waitNum>0 and waitList[0,4]<=currentTime: # 所调度的作业必须已
            tempNum=0
            for i in range(len(result)): 
                if result[i][10]>=currentTime:
                    tempNum+=result[i][6]
            accessNodeNum=nodeNum-tempNum
            # print("可用cpu数目：",accessNodeNum)
            if waitList[0,6]<=accessNodeNum: # 有可用节点
                 temp_result=list(waitList[0,:])                    # 0-8
                 temp_result.append(currentTime)                    # 9实际开始时间
                 temp_result.append(currentTime+cpuTime[0])         # 10实际结束时间
                 temp_result.append(currentTime-waitList[0,4])      # 11延迟时间
                 #temp_result.append(fassibleIndex[0:waitList[0,6]]) # 12所分配的节点
                 temp_result.append(0)  # 12无意义
                 result.append(temp_result)
                 maxFinish=max([maxFinish,currentTime+cpuTime[0]])
                 waitList=np.copy(waitList[1:,:]) # 将已开始执行的任务删除
                 cpuTime=np.copy(cpuTime[1:])
                 waitNum-=1 # 更新等待作业队列数目
            else: # 时间继续进行
                break # 没有满足数目的节点
        # 更新列表中的其他作业信息
        if len(waitList)==0:
            break
        
        finishTime=0 # 计算至当前时刻，用户所有所用的时间
        for i in userType:
            for value in result:
                if value[8]==i: # 同一用户
                    finishTime+=currentTime-value[9]
            sameUser=1+np.where(waitList[1:,8]==i)[0]
            waitList[sameUser,3]=finishTime
       
        waitList[:,7]=currentTime-waitList[:,4] # 更新等待时长
        errIndex=np.where(waitList[:,7]<0)
        waitList[errIndex,7]=0
        insex=np.where(waitList[:,4]<=currentTime)[0] # 查看有没有在当前时刻未被处理的作业
        if len(insex)>0:
            currentTime+=1
        else:
            currentTime=waitList[0,4]
    print("FCFS的结果统计：")
    evaluate(result) # 计算评价指标

def myPriority(nodeNum,waitList,allCpuTime,weight):
    currentTime=0  # 当前时刻, 以第一个作业提交时刻为初始0点
    waitNum=waitList.shape[0]      # 等待列表作业数目
    # 计算当前阶段的优先级
    userType=np.unique(waitList[:,8]) # 用户类型
    result,maxFinish=[],0
    currentList=[] # 当前时刻作业列表
    cpuTime,waitIndex=[],np.ones((1,waitNum),dtype=int)
    while waitNum:
        print("当前时刻：",int(currentTime)," 剩余数目：",waitNum," 已完成",len(result),"个")
        # 判断该时刻有没有新来的作业，有则添加
        index=np.where(waitList[:,4]==currentTime)[0]
        if len(index)>0:
            waitIndex[0,index]=0
            for i in range(len(index)):
                currentList.append(waitList[i,:])
                cpuTime.append(allCpuTime[i])
                
        if len(currentList)>0:# 如果排队队列中有等待作业
            tempNum=0
            for i in range(len(result)): 
                if result[i][10]>=currentTime:
                    tempNum+=result[i][6]
            accessNodeNum=nodeNum-tempNum # 当前时刻所剩余cpu数目
        
            # 计算等待队列中的作业优先级
            priority=currentPriority2(currentList,accessNodeNum,weight)
            sortIndex=np.argsort(-priority) # 获取降序索引
            currentList=np.array(currentList)[sortIndex,:]
            cpuTime=np.array(cpuTime)[sortIndex]
            
            while len(currentList)>0: # 队列有作业        
                if currentList[0,6]<=accessNodeNum: # 节点够用
                    temp_result=list(currentList[0,:])                       # 0-8
                    temp_result.append(currentTime)                    # 9实际开始时间
                    temp_result.append(currentTime+cpuTime[0])                 # 10实际结束时间
                    temp_result.append(currentTime-waitList[0,4])      # 11延迟时间
                    temp_result.append(0)  # 12无意义
                    result.append(temp_result)
                    
                    # 更新被抢占次数
                    stopIndex=1+np.where(currentList[1:,4]<=currentList[0,4])[0]
                    currentList[stopIndex,5]+=1 # 时间提的早却没分上
                    accessNodeNum-=currentList[0,6]

                    currentList=np.copy(currentList[1:,:]) # 将已开始执行的任务删除
                    cpuTime=np.copy(cpuTime[1:])
                    waitNum-=1 # 更新等待作业队列数目
                else: # 时间继续进行
                    break # 没有满足数目的节点
                    
        if len(currentList)>0:
            currentTime+=1
            finishTime=0 # 计算至当前时刻，用户所有所用的时间
            for i in userType:
                for value in result:
                    if value[8]==i: # 同一用户
                        finishTime+=currentTime-value[9]
                    sameUser=1+np.where(currentList[1:,8]==i)[0]
                    currentList[sameUser,3]=finishTime
            currentList[:,7]=currentTime-currentList[:,4] # 更新等待时长
            currentList[np.where(currentList[:,7]<0),7]=0
            currentList,cpuTime=list(currentList),list(cpuTime)
        else:
            currentList,cpuTime,fistOnesIndex=[],[],np.where(waitIndex[0,:]==1)[0]
            if len(fistOnesIndex)>0:
                currentTime= int(waitList[fistOnesIndex[0],4]) # 时间跳转
            else:
                currentTime+=1
    print("myPrioriy的结果统计：")           
    evaluate(result) # 计算评价指标
    
