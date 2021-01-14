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



def evaluate(result): # 计算评价指标
    res=trans(result)
    indicator1 = np.std(res[:,11])              # 等待时长的均衡性
    indicator2 = np.max(res[:,11])              # 最大延迟时长
    indicator3 = np.mean(res[:,11])             # 平均等待时长
    indicator4 = res.shape[0]/np.max(res[:,10]) # 系统的吞吐率
    indicator5 = np.mean(res[:,10]-res[:,4])    # 平均响应时间
    indicator6 = np.mean((res[:,10]-res[:,4])/(res[:,10]-res[:,9])+1) # 平均减速
    print("等待时长的均衡性：",indicator1)
    print("最大延迟时长：",indicator2)
    print("平均等待时长：",indicator3)
    print("系统的吞吐率：",indicator4)
    print("平均响应时间：",indicator5)
    print("平均减速：",indicator6)
    #return np.array([indicator1,indicator2,indicator3,indicator4,indicator5,indicator6])

