#构建数据集
def loadDataSet():
    return [[1 ,3 ,4] ,[2 ,3 ,5] ,[1 ,2 ,3 ,5] ,[2 ,5]]

def createC1(dataSet):    #1项集
    '''
        构建初始候选项集的列表，即所有候选项集只包含一个元素，C1是大小为1的所有候选项集的集合
    '''
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()   #排序
    return list(map(frozenset ,C1))

#扫描模块
def scanD(D ,CK ,minSupport):
    '''
    计算Ck中的项集在数据集合D(记录或者transactions)中的支持度,
    返回满足最小支持度的项集的集合，和所有项集支持度信息的字典。
    '''
    ssCnt = {}  #字典
    for tid in D:
        # 对于每一条transaction
        for can in CK:
            # 对于每一个候选项集can，检查是否是transaction的一部分# 即该候选can是否得到transaction的支持
            if can.issubset(tid):  #是否为子集
                if not can in ssCnt:
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1
    numItems = float(len(list(D)))
    retlist = []
    supportData = {}
    for key in ssCnt:
        # 每个项集的支持度
        support = ssCnt[key ] /numItems
        if support >= minSupport:
            # 将满足最小支持度的项集，加入retList
            retlist.insert(0 ,key)
        # 汇总支持度数据
        supportData[key] = support
    return retlist ,supportData

# Aprior算法
def aprioriGen( Lk, k ):
    '''
    由初始候选项集的集合Lk生成新的生成候选项集，
    k表示生成的新项集中所含有的元素个数
    '''
    retList = []
    lenLk = len(Lk)
    for i in range(lenLk):
        for j in range( i + 1, lenLk ):
            L1 = list( Lk[ i ] )[ : k - 2 ]
            L2 = list( Lk[ j ] )[ : k - 2 ]
            L1.sort();L2.sort()
            if L1 == L2:
                retList.append( Lk[ i ] | Lk[ j ] ) 
    return retList

def apriori( dataSet, minSupport = 0.5 ):
    # 构建初始候选项集C1
    C1 = createC1(dataSet)
    
    # 将dataSet集合化，以满足scanD的格式要求
    D = dataSet # map( set, dataSet )
    
    # 构建初始的频繁项集，即所有项集只有一个元素
    L1, suppData = scanD( D, C1, minSupport )
    L = [ L1 ]
    # 最初的L1中的每个项集含有一个元素，新生成的
    # 项集应该含有2个元素，所以 k=2
    k = 2
    
    while ( len( L[k -2] ) > 0 ):
        Ck = aprioriGen( L[ k - 2 ], k )
        Lk, supK = scanD(D, Ck, minSupport )
        
        # 将新的项集的支持度数据加入原来的总支持度字典中
        suppData.update(supK)
        
        # 将符合最小支持度要求的项集加入L
        L.append(Lk)
        
        # 新生成的项集中的元素个数应不断增加
        k += 1
    # 返回所有满足条件的频繁项集的列表，和所有候选项集的支持度信息
    return L, suppData



# 规则生成与评价   
def calcConf( freqSet, H, supportData, brl, minConf=0.7 ):
    '''
    计算规则的可信度，返回满足最小可信度的规则。
    
    freqSet(frozenset):频繁项集
    H(frozenset):频繁项集中所有的元素
    supportData(dic):频繁项集中所有元素的支持度
    brl(tuple):满足可信度条件的关联规则
    minConf(float):最小可信度
    '''
    prunedH = []
    for conseq in H:
        conf = supportData[freqSet] / supportData[freqSet - conseq]
        if conf >= minConf:
            print(freqSet - conseq, '-->', conseq, 'conf:', conf)
            brl.append( ( freqSet - conseq, conseq, conf ) )
            prunedH.append(conseq)
    return prunedH

def rulesFromConseq(freqSet, H, supportData, brl, minConf=0.7):
    '''
    对频繁项集中元素超过2的项集进行合并。
    
    freqSet(frozenset):频繁项集
    H(frozenset):频繁项集中的所有元素，即可以出现在规则右部的元素
    supportData(dict):所有项集的支持度信息
    brl(tuple):生成的规则
    
    '''
    m = len( H[0] )
    
    # 查看频繁项集是否大到移除大小为 m　的子集
    if len(freqSet) > m + 1:
        Hmp1 = aprioriGen(H, m+1)
        Hmp1 = calcConf( freqSet, Hmp1, supportData, brl, minConf )
        
        # 如果不止一条规则满足要求，进一步递归合并
        if len( Hmp1 ) > 1:
            rulesFromConseq( freqSet, Hmp1, supportData, brl, minConf )

def generateRules(L, supportData, minConf=0.7):
    '''
    根据频繁项集和最小可信度生成规则。
    
    L(list):存储频繁项集
    supportData(dict):存储着所有项集（不仅仅是频繁项集）的支持度
    minConf(float):最小可信度
    '''
    bigRuleList = []
    for i in range( 1, len( L ) ):
        for freqSet in L[ i ]:
            # 对于每一个频繁项集的集合freqSet
            H1 = [ frozenset( [ item ] ) for item in freqSet ]
            
            # 如果频繁项集中的元素个数大于2，需要进一步合并
            if i > 1:
                rulesFromConseq( freqSet, H1, supportData, bigRuleList, minConf )
            else:
                calcConf(freqSet, H1, supportData, bigRuleList, minConf)
    return bigRuleList

if __name__ == '__main__':
    # 导入数据集
    dataSet = loadDataSet()
    L, suppData = apriori(dataSet, 0.5 )
    rules = generateRules( L, suppData, minConf=0.7 )
    print('rules:\n', rules)
    
    