import numpy as np
import ScheduleAlgirithm as SA
import TaskSchedule as TS
import multiprocessing as mp

def func(nodeNum,waitList,allCpuTime,weight,num=20):
    # x输入粒子位置  y 粒子适应度值
    result=SA.myPriority(nodeNum,waitList,allCpuTime,weight)
    res=TS.trans(result)
    indicator1 = np.std(res[:,11])              # 等待时长的均衡性(方差)
    indicator2 = np.mean(res[:,11])             # 平均等待时长
    indicator3 = 0 # 排队的情况
    n=np.shape(res)[0] # 作业总数
    m=n//num
    for i in range(m):
        allType=np.unique(res[i*num:(i+1)*num,8])# 用户总类别
        a=np.zeros((len(allType),1))
        for typ in  range(len(allType)):
            a[typ,0]=len(np.where(res[i*num:(i+1)*num,8]==allType[typ])[0])
        indicator3+=np.std(a)
    return indicator1+indicator2+indicator3

def initpopvfit(nodeNum,waitList,allCpuTime,sizepop): # 初始化种群
    pop=np.random.random((sizepop,7))
    v=-0.5+np.random.random((sizepop,7))
    fitness = np.zeros((sizepop,1))

    for i in range(sizepop):
        print("     正在计算第",i,"个粒子的适应度值")
        fitness[i,0] = func(nodeNum,waitList,allCpuTime,pop[i,:])
    return pop,v,fitness

def getinitbest(fitness,pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop,gbestfitness = pop[fitness.argmax()].copy(),fitness.max()
    #个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
    pbestpop,pbestfitness = pop.copy(),fitness.copy()
    
    return gbestpop,gbestfitness,pbestpop,pbestfitness  



def PSO_main(w,lr,maxgen,sizepop,rangepop,rangespeed,waitList,allCpuTime,nodeNum):
    print("正在初始化种群....")
    pop,v,fitness = initpopvfit(nodeNum,waitList,allCpuTime,sizepop)
    gbestpop,gbestfitness,pbestpop,pbestfitness = getinitbest(fitness,pop)
    
    result = np.zeros((maxgen,8))
    for i in range(maxgen):
        #速度更新
        print("第",i,"代...")
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
        for j in range(sizepop):
            fitness[j] = func(nodeNum,waitList,allCpuTime,pop[j,:])
            
        for j in range(sizepop):
            if fitness[j] > pbestfitness[j]:
                pbestfitness[j] = fitness[j]
                pbestpop[j] = pop[j].copy()
        
        if pbestfitness.max() > gbestfitness :
            gbestfitness = pbestfitness.max()
            gbestpop = pop[pbestfitness.argmax()].copy()
        
        result[i,0:7] = gbestpop # 最优参数
        result[i,7] = gbestfitness   # 最优参数对应的值
    return result

