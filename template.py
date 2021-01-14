import numpy as np
import ScheduleAlgirithm as SA
import TaskSchedule as TS
import multiprocessing as mp
import matplotlib.pyplot as plt

def func(nodeNum,waitList,allCpuTime,weight):
    # x输入粒子位置  y 粒子适应度值
    result=SA.myPriority(nodeNum,waitList,allCpuTime,weight)
    res=TS.trans(result)
    return np.max(res[:,11]) # 以最大等待时长最小化为目标

def initpopvfit(nodeNum,waitList,allCpuTime,sizepop): # 初始化种群
    pop=np.random.random((sizepop,7))
    v=-0.5+np.random.random((sizepop,7))
    fitness = np.zeros((sizepop,1))

    for i in range(sizepop):
        fitness[i,0] = func(nodeNum,waitList,allCpuTime,pop[i,:])
    return pop,v,fitness

def getinitbest(fitness,pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop,gbestfitness = pop[fitness.argmax()].copy(),fitness.max()
    #个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
    pbestpop,pbestfitness = pop.copy(),fitness.copy()
    
    return gbestpop,gbestfitness,pbestpop,pbestfitness  


if __name__ == '__main__':
    w = 1
    lr = (0.49445,1.49445)
    maxgen = 100
    sizepop = 20
    rangepop = (0,1)
    rangespeed = (-0.5,0.5)
    
    dataString="2020-04-30" # 使用数据集
    waitList,allCpuTime = TS.dataPross(dataString)
    nodeNum=max(300,np.max(waitList[:,6])+100)
    
    # 初始化种群
    pop=np.random.random((sizepop,7))
    v=-0.5+np.random.random((sizepop,7))
    fitness = np.zeros((sizepop,1))
    # 并行计算适应度值
    q=mp.Queue()
    s=[]
    for i in range(sizepop):
        s.append(mp.Process(target=func,args=(q,nodeNum,waitList,allCpuTime,pop[i,:])))
    for i in range(sizepop):
        s[i].start()
    for i in range(sizepop):
        s[i].join()
    for i in range(sizepop):
        fitness[i,0]=q.get()
   
    gbestpop,gbestfitness,pbestpop,pbestfitness = getinitbest(fitness,pop)
    
    result = np.zeros((maxgen,8))
    for i in range(maxgen):
        #速度更新
        for j in range(sizepop):
            v[j] += lr[0]*np.random.rand()*(pbestpop[j]-pop[j])+lr[1]*np.random.rand()*(gbestpop-pop[j])
        v[v<rangespeed[0]] = rangespeed[0]
        v[v>rangespeed[1]] = rangespeed[1]
        
        #粒子位置更新
        for j in range(sizepop):
            pop[j] += 0.5*v[j]
        pop[pop<rangepop[0]] = rangepop[0]
        pop[pop>rangepop[1]] = rangepop[1]
        
        #适应度更新
        q2=mp.Queue()
        s=[]
        for j in range(sizepop):
            s.append(mp.Process(target=func,args=(q2,nodeNum,waitList,allCpuTime,pop[j,:])))
        for j in range(sizepop):
            s[j].start()
        for j in range(sizepop):
            s[j].join()
        for j in range(sizepop):
            fitness[j,0]=q2.get() 
            
            
        for j in range(sizepop):
            if fitness[j] > pbestfitness[j]:
                pbestfitness[j] = fitness[j]
                pbestpop[j] = pop[j].copy()
        
        if pbestfitness.max() > gbestfitness :
            gbestfitness = pbestfitness.max()
            gbestpop = pop[pbestfitness.argmax()].copy()
        
        result[i,0:7] = gbestpop # 最优参数
        result[i,7] = gbestfitness   # 最优参数对应的值
    plt.plot(result[:,7])
    plt.show()
