# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 09:57:42 2020

@author: Wang
"""


import numpy as np
import pandas as pd

def dataPross(dataString):
    data=pd.read_csv("C:/Users/Wang/Desktop/data.csv") #从CSV文件导入数据
    data["startTime"]=pd.to_datetime(data["startTime"])
    data = data.sort_values(by="startTime") # 拍提交时间排序
    data = data.set_index("startTime") # 将date设置为index
    userData=data[dataString]        # 获取5月8日的数据
    userData["startTime"]=userData.index
    userData =userData.set_index("id") # 将作业id设置为index
    userData["userRank"]=1     # 用户等级
    userData["submisTime"]=1   # 用户提交次数
    userData["haveUsedTime"]=0 # 已使用时长
    userData["backdTime"]=0    # 被抢占个数
    userData["havewaitTime"]=0 # 等待时长
    print(userData.shape)
    userType=pd.DataFrame(pd.unique(userData["username"]),columns={'username'})
    userType["userClass"]=userType.index
    userType =userType.set_index("username") # 将作业id设置为index
    df=userType.loc[userData["username"]]
    df.index=userData.index
    pd.concat([userData,df],axis=1)
    userData["userClass"]=userType.loc[userData["username"]].values
    userData["startTimeTrans"] = np.round(pd.DataFrame(userData['startTime'] -userData['startTime'].iloc[0]).values/np.timedelta64(1, 's'))
    
    # 0作业ID 1用户等级 2用户提交次数 3已使用时长 4作业提交时刻 
    # 5被抢占个数 6所需节点 7等待时长 8所属用户
    waitList=pd.DataFrame([userData.index,userData["userRank"],userData["submisTime"], \
                           userData["haveUsedTime"],userData["startTimeTrans"],userData["backdTime"], \
                           userData["numCores"],userData["havewaitTime"],userData["backdTime"]
                          ]).values.T#
    cpuTime=userData["cpuSecs"].values # 作业的cpu使用时间
    return waitList,cpuTime

def trans(result): # 将列表转为矩阵
    m,n=len(result),len(result[0])
    res=np.zeros((m,n-1))
    for i in range(m):
        for j in range(n-1):
            res[i,j]=result[i][j]
    return res

def currentPriority(waitList,currentTime,accessNodeNum,weight,waitNum):
    # 计算列表中作业的优先级
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

def currentPriority2(currentList,accessNodeNum,weight):
    # 计算列表中作业的优先级
    compareList=trans(currentList) 
    priority=weight[0]*compareList[:,1]/(1+sum(compareList[:,1]))+\
                 weight[1]*(1-compareList[:,2]/max(compareList[:,2]))+\
                 weight[2]*compareList[:,3]/(1+sum(compareList[:,3]))+\
                 weight[3]*(1-compareList[:,4]/(1+max(compareList[:,4])))+\
                 weight[4]*compareList[:,5]/(1+sum(compareList[:,5]))+\
                 weight[5]*(1-(abs(compareList[:,6]-accessNodeNum)/max(abs(compareList[:,6]-accessNodeNum))))+\
                 weight[6]*compareList[:,7]/(1+sum(compareList[:,7]))   
    return priority

def caculation_priority(currentList,algirithmName,accessNodeNum=0,weight=None):
    compareList=trans(currentList) 
    if algirithmName=="FCFS":
        priority=1/(1+np.argsort(compareList[:,4])) # 按时间算优先级
    elif algirithmName=="myPriority":
        priority=weight[0]*compareList[:,1]/(1+sum(compareList[:,1]))+\
                 weight[1]*(1-compareList[:,2]/max(compareList[:,2]))+\
                 weight[2]*compareList[:,3]/(1+sum(compareList[:,3]))+\
                 weight[3]*(1-compareList[:,4]/(1+max(compareList[:,4])))+\
                 weight[4]*compareList[:,5]/(1+sum(compareList[:,5]))+\
                 weight[5]*(1-(abs(compareList[:,6]-accessNodeNum)/max(abs(compareList[:,6]-accessNodeNum))))+\
                 weight[6]*compareList[:,7]/(1+sum(compareList[:,7]))  
    return priority

def evaluate(result): # 计算评价指标
    res=trans(result)
    indicator1 = np.std(res[:,11])              # 等待时长的均衡性
    indicator2 = np.max(res[:,11])              # 最大延迟时长
    indicator3 = np.mean(res[:,11])             # 平均等待时长
    indicator4 = res.shape[0]/np.max(res[:,10]) # 系统的吞吐率
    indicator5 = np.mean(res[:,10]-res[:,4])    # 平均响应时间
    indicator6 = np.mean((res[:,10]-res[:,4])/(res[:,10]-res[:,9]+1)) # 平均减速
    print("等待时长的均衡性：",indicator1)
    print("最大延迟时长：",indicator2)
    print("平均等待时长：",indicator3)
    print("系统的吞吐率：",indicator4)
    print("平均响应时间：",indicator5)
    print("平均减速：",indicator6)
    print("最后完成时刻：",np.max(res[:,10]))
    #return np.array([indicator1,indicator2,indicator3,indicator4,indicator5,indicator6])

def plotTake(result,currentTime,nodeNum,control=3):# 作当前时刻的调度图示
    import matplotlib.pyplot as plt
    import numpy as np
    color,n=['r','g','b','m'],10
    if control==3:
        plt.subplot(211)
    if control==1 or control==3:
        for i in range(len(result)):
            for j in result[i][12]:
                if result[i][10]<=currentTime:
                    X = np.linspace(result[i][9], result[i][10], n, endpoint=True)
                    plt.text(np.max([result[i][9],(result[i][10]+result[i][9])/2]),j+0.5,str(result[i][0]))
                else:
                    X = np.linspace(result[i][9], currentTime, n, endpoint=True)
                    plt.text(np.max([result[i][9],(currentTime+result[i][9])/2]),j+0.5,str(result[i][0]))
                y1,y2=j*np.ones((1,n)),np.ones((1,n))*(j+1)
                plt.fill_between(X,y1[0],y2[0],facecolor=color[i%4])

        plt.plot([currentTime,currentTime],[-0.5,nodeNum+0.5],'k')
        plt.xlabel('Time')
        plt.ylabel('Node')
    plt.title('Task Scheduling')
    
    if control==3:
        plt.subplot(212)
    if control==2 or control==3:
        for i in range(len(result)):
            for j in result[i][12]:
                # plt.plot(np.array([result[i][9],result[i][10]]),np.array([j,j]),color[i%4])
                # plt.plot(np.array([result[i][9],result[i][9]]),np.array([j,j+1]),color[i%4])
                # plt.plot(np.array([result[i][9],result[i][10]]),np.array([j+1,j+1]),color[i%4])
                # plt.plot(np.array([result[i][10],result[i][10]]),np.array([j,j+1]),color[i%4])
                # plt.text(np.max([result[i][9],(result[i][10]+result[i][9])/2]),j+0.5,str(result[i][0]))
        
                X = np.linspace(result[i][9], result[i][10], n, endpoint=True)
                plt.text(np.max([result[i][9],(result[i][10]+result[i][9])/2]),j+0.5,str(result[i][0]))
                y1,y2=j*np.ones((1,n)),np.ones((1,n))*(j+1)
                plt.fill_between(X,y1[0],y2[0],facecolor=color[i%4])

        plt.plot([currentTime,currentTime],[-0.5,nodeNum+0.5],'k')
        plt.xlabel('Time')
        plt.ylabel('Node')
    plt.show()

def appendResult(index,result,currentList,accessNodeNum,currentTime,cpuTime,waitNum):
    temp_result=list(currentList[index,:])                 # 0-8
    temp_result.append(currentTime)                    # 9实际开始时间
    temp_result.append(currentTime+cpuTime[index])         # 10实际结束时间
    temp_result.append(currentTime-currentList[index,4])      # 11延迟时间
    temp_result.append(0)  # 12无意义
    result.append(temp_result)
    
    # 更新被抢占次数
    stopIndex=np.where(currentList[:,4]<=currentList[index,4])[0]
    currentList[stopIndex,5]+=1 # 时间提的早却没分上
    accessNodeNum-=currentList[index,6]
    if index==0:
        currentList=np.copy(currentList[1:,:]) # 将已开始执行的任务删除
        cpuTime=np.copy(cpuTime[1:])    
    else:
        currentList=np.vstack((currentList[:index,:],currentList[index+1:,:])) # 将已开始执行的任务删除
        cpuTime=np.hstack((cpuTime[:index],cpuTime[index+1:]))
    waitNum-=1 # 更新等待作业队列数目
    return result,temp_result,currentList,cpuTime,waitNum,accessNodeNum