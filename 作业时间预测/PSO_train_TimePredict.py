# -*- coding: utf-8 -*-
"""
Created on Fri Jun 19 09:24:38 2020

@author: Wang
"""


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime


def dataRead(fileName):
    print("正在读取与处理数据...")
    data=pd.read_csv(fileName).iloc[:,[2,3,6,7,8,9,16]]    # '客户端', '用户', '超算', 'software','version', 'numCores','cpuSecs'
    client_type=data['clientId'].unique()    # 客户端类型
    user_type=data['username'].unique()      # 用户类型
    sc_type=data['serverID'].unique()        # 超算类型
    software_type=data['software'].unique()  # 软件类型
    version_type=data['version'].unique()    # 版本类型
    for i in range(len(client_type)):
        data['clientId'].loc[data['clientId']==client_type[i]]=i
    for i in range(len(user_type)):
        data['username'].loc[data['username']==user_type[i]]=i
    for i in range(len(sc_type)):
        data['serverID'].loc[data['serverID']==sc_type[i]]=i
    for i in range(len(software_type)):
        data['software'].loc[data['software']==software_type[i]]=i
    for i in range(len(version_type)):
        data['version'].loc[data['version']==version_type[i]]=i
   
    preData=[]
    for i in range(len(software_type)):
        obj=np.array(data.loc[data['software'].isin([i])])[:,(0,1,2,4,5,6)]
        obj=obj[obj[:,-1]>=60,:] # 运行时长少于60秒的剔除
        if np.shape(obj)[0]>=100: # 数据量少于100的剔除
            preData.append(obj) 
    return preData 

def dataAnalyze(fileName):
    print("正在读取与处理数据...")
    data=pd.read_csv(fileName).iloc[:,[2,3,6,7,8,9,16]]    # '客户端', '用户', '超算', 'software','version', 'numCores','cpuSecs'
    software_type=data['software'].unique()  # 软件类型
    
    preData=np.zeros(len(software_type),dtype=int)
    for i in range(len(software_type)):
       preData[i]=len(np.where(data['software']==software_type[i])[0])
        
    plt.barh(range(len(software_type)), preData,color='rgb',tick_label=software_type) 
    for a,b in zip(preData,range(len(software_type))):
        plt.text(a+1000, b, '%' % preData[i], ha='center', va= 'bottom',fontsize=9)
    plt.xlabel("error")
    plt.ylabel("percentage")
    plt.show()  

    
    
def dataSplit(preData,train_rate):
    print("正在分割测试数据与训练数据...")
    
    trainData,testData=[],[]     
    for i in range(len(preData)):
        data=preData[i]
        if np.shape(data)[0]<=5:
            continue
        np.random.seed(0) # 使用相同的数据集
        train_num,index=int(train_rate*data.shape[0]),np.arange(data.shape[0])
        trainData.append(data[index[0:train_num],:])
        testData.append(data[index[train_num:],:])
    return trainData,testData

def knnFit(trainData,testData,K,w):
    m, predict,len_w = len(trainData),[],len(w)
    for s in range(m):
        test_data,train_data=testData[s],trainData[s]
        test_num,n,train_num=np.shape(test_data)[0],np.shape(test_data)[1],np.shape(train_data)[0]
        for i in range(test_num):   
            dis=np.zeros(train_num,dtype=float) # 计算距离
            for j in range(n-1):
                dis+=w[j]*((test_data[i,j]-train_data[:,j])!=0) # 单个测试个体与所有训练个体之间的距离
            coefficient=(1-(1/K+np.sort(dis)[:K])/(1+np.sum(np.sort(dis)[:K])))      # 相似性越高，采用的比例越大
            #coefficient=(1-(1/K+np.sort(dis)[:K])/(1+np.sum(np.sort(dis)[:K])))*np.array(w[n-1:len_w-1])      # 相似性越高，采用的比例越大

            predict.append(w[len_w-1]*np.sum(train_data[np.argsort(dis)[:K],-1])+np.dot(coefficient,train_data[np.argsort(dis)[:K],-1])) # 采用均值
           
    return np.array(predict)

def evaluate(testData,predict):
    test_data=np.vstack(testData)
    test_num=np.shape(test_data)[0]
    dif_index=np.argsort(abs(test_data[:,-1]-predict))
    result=np.zeros((test_num,4),dtype='float')
    result[:,0]=test_data[dif_index,-1][:] # 测试数据
    result[:,1]=predict[dif_index]         # 预测数据
    result[:,2]=predict[dif_index]-test_data[dif_index,-1]
    result[:,3]=abs(result[:,2])/result[:,0]
    res=pd.DataFrame(result)
    res.columns=['实际值','预测值','差值','误差率']
    return res

def statistics(res):
    result=np.array(res)
    num,num_list,name_list=3,[],[]
    for i in range(1,2*num+1):
        num_list.append(np.sum(abs(result[:,2])/result[:,0]<(i*0.05))/np.shape(result)[0]) # 成功率
        name_list.append(str(i*5)+'%')
    plt.bar(range(len(num_list)), num_list,color='rgb',tick_label=name_list) 
    plt.xlabel("error")
    plt.ylabel("percentage")
    plt.show()  
    
def calculate_fitness(x,trainData,testData,K):
    m=np.shape(x)[0]
    res=np.zeros(m)
    for i in range(m):
        print("  正在计算第",i,"个个体的适应度值....")
        predict=knnFit(trainData,testData,K,x[i,:])
        temp=np.array(evaluate(testData,predict))
        res[i]=(np.sum(temp[temp[:,2]<0,2]**2)+np.sum(temp[temp[:,2]>=0,2]))/1000000# 整体的误差最小
    return res

def PSO(K,max_steps,population_size,w,c1,c2,dim,x_bound,v_bound,trainData,testData):
    # step1: 产生初始种群
    x = np.random.uniform(x_bound[0], x_bound[1],(population_size, dim))  # 初始化粒子群位置
    v = np.random.rand(population_size, dim)                              # 初始化粒子群速度
    
    # step2: 计算初始适应度值
    print("初始化种群与计算初始适应度值....")
    fitness = calculate_fitness(x,trainData,testData,K)
    p,pg = x, x[np.argmin(fitness)]                                        # 个体的最佳位置全局最佳位置
    individual_best_fitness,global_best_fitness = fitness,np.min(fitness)  # 个体的最优适应度 全局最佳适应度
    res=np.zeros((max_steps,dim+1))
    
    # step3: 迭代寻优
    for step in range(max_steps):
        print("正在进行第",step,"代迭代....")
        starttime0= datetime.datetime.now()
    
        r1 = np.random.rand(population_size, dim)
        r2 = np.random.rand(population_size, dim)
    
        # 更新速度和权重 边界处理
        v = w*v+c1*r1*(p-x)+c2*r2*(pg-x)
        v[v<v_bound[0]]=v_bound[0]
        v[v>v_bound[1]]=v_bound[1]
    
        x = v + x
        x[x<x_bound[0]]=x_bound[1]
        x[x>x_bound[1]]=x_bound[1]
    
        fitness = calculate_fitness(x,trainData,testData,K)
    
        # 需要更新的个体
        update_id = np.greater(individual_best_fitness, fitness)
        p[update_id] = x[update_id]
        individual_best_fitness[update_id] = fitness[update_id]
        
        # 新一代出现了更小的fitness，所以更新全局最优fitness和位置
        if np.min(fitness) < global_best_fitness:
            pg = x[np.argmin(fitness)]
            global_best_fitness = np.min(fitness)
        
        res[step,0]=global_best_fitness
        res[step,1:]=pg
        print("  本代运行时间",(datetime.datetime.now() - starttime0).seconds,"秒")
    
    # step4: 结果展示
    plt.plot(res[:,0])
    plt.title("Evolution curve")
    plt.xlabel("weight")
    plt.ylabel("std_value")
    plt.show()
    return res


#def main():
train_rate,K,fileName=0.8,3,"C:/Users/Wang/my_code/作业时间预测/data.csv" # 数据读取
max_steps,population_size,w,c1,c2,dim,x_bound,v_bound=100,10,0.6,2,2,6,[0,1],[-1,1]

starttime= datetime.datetime.now() # 计时

trainData,testData=dataSplit(dataRead(fileName),train_rate) # 数据处理与划分
print("数据处理时间",(datetime.datetime.now() - starttime).seconds,"秒")
   
res=PSO(K,max_steps,population_size,w,c1,c2,dim,x_bound,v_bound,trainData,testData) # 优化参数
print("最优参数:",res[-1,1:],"\n标准差：",res[-1,0])

predict=knnFit(trainData,testData,K,res[-1,1:])
result=evaluate(testData,predict)
statistics(result) # 结果统计

print("整个程序运行",(datetime.datetime.now() - starttime).seconds,"秒")




