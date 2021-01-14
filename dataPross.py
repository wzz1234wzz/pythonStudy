# -*- coding: utf-8 -*-
"""
Created on Sat May  9 10:59:52 2020

@author: Wang
"""


import pandas as pd
import numpy as np

data=pd.read_csv("C:/Users/Wang/Desktop/data.csv") #从CSV文件导入数据
data["startTime"]=pd.to_datetime(data["startTime"])
data = data.sort_values(by="startTime") # 拍提交时间排序
data = data.set_index("startTime") # 将date设置为index
userData=data["2020-04-30"]        # 获取5月8日的数据
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

    
    
    
    
    
    
    
    
    