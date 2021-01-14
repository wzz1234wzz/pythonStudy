# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 13:07:09 2020
作业时间预测
@author: Wang
"""
import pandas as pd
import numpy as np

def dataRead(fileName):
    data=pd.read_csv(fileName).iloc[:,[2,3,6,7,8,9,16]]    # '客户端', '用户', '超算', 'software','version', 'numCores','cpuSecs'
    # return pd.read_csv(fileName,encoding = 'gb18030').iloc[:,[2,3,6,7,8,9,16]]    # '客户端', '用户', '超算', 'software','version', 'numCores','cpuSecs'
    client_type=data['clientId'].unique()    # 客户端类型
    user_type=data['username'].unique()      # 用户类型
    sc_type=data['serverID'].unique()        # 超算类型
    software_type=data['software'].unique()  # 软件类型
    version_type=data['version'].unique()    # 版本类型
    preData=[]
    for i in range(data.shape[0]):
        temp=client_type==data.iloc[i,0]
        data.iloc[i,0]=np.where(temp==True)[0][0]
        temp=user_type==data.iloc[i,1]
        data.iloc[i,1]=np.where(temp==True)[0][0]
        temp=sc_type==data.iloc[i,2]
        data.iloc[i,2]=np.where(temp==True)[0][0]
        temp=software_type==data.iloc[i,3]
        data.iloc[i,3]=np.where(temp==True)[0][0]
        temp=version_type==data.iloc[i,4]
        data.iloc[i,4]=np.where(temp==True)[0][0]
    for i in range(len(software_type)):
        trydata=np.array(data.loc[data['software'].isin([i])])[:,(0,1,2,4,5,6)]
        if np.shape(trydata)[0]>1:
            preData.append(trydata)
    return preData 
    
def dataSplit(data,train_rate):
    train_num,index=int(train_rate*data.shape[0]),np.arange(data.shape[0])
    np.random.shuffle(index)
    return data[index[0:train_num],:],data[index[train_num:],:]

def knnFit(train_data,test_data,K,w,):
    test_num,n,train_num=np.shape(test_data)[0],np.shape(test_data)[1],np.shape(train_data)[0]
    predict=np.zeros(test_num) # 记录
    for i in range(test_num):   
        dis=np.zeros(train_num) # 计算距离
        for j in range(n-1):
            dis+=w[j]*(test_data[i,j]-train_data[:,j]==0)
        predict[i]=np.mean(train_data[np.argsort(dis)[:K],-1])
    return predict

def evaluate(test_data,predict):
    test_num=np.shape(test_data)[0]
    dif_index=np.argsort(abs(test_data[:,-1]-predict))
    result=np.zeros((test_num,3),dtype='float')
    result[:,0]=test_data[dif_index,-1][:]
    result[:,1]=predict[dif_index]
    result[:,2]=abs(test_data[dif_index,-1]-predict[dif_index])
    return result

def main(fileName,train_rate,K,w):
    preData,res =dataRead(fileName), []      # 数据读取与处理
    for i in range(len(preData)):
        data=preData[i]
        train_data,test_data=dataSplit(data, train_rate)                  # 数据分割
        predict=knnFit(train_data, test_data, K, w)                       # 训练模型
        res.append(evaluate(test_data, predict))
        print(len(res))
    res=np.vstack(res)
    std=np.std(res[:,2])
    #print("result:\n",res,'\nstd:',std)
    return std

def fit(preData,train_rate,K,w):
     for i in range(len(preData)):
        data=preData[i]
        train_data,test_data=dataSplit(data, train_rate)                  # 数据分割
        predict=knnFit(train_data, test_data, K, w)                       # 训练模型
        res.append(evaluate(test_data, predict))
        print(len(res))
    res=np.vstack(res)
    std=np.std(res[:,2])
    #print("result:\n",res,'\nstd:',std)
    return std
    
if __name__=='__main__':
    train_rate,K,w,fileName=0.8,5,[0.1,0.5,0.1,0.1,1],"C:/Users/Wang/my_code/作业时间预测/data.csv" # 数据读取
    main(fileName,train_rate,K,w)
    
    