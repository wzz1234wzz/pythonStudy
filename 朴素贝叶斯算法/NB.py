import numpy as np 
from math import log  

def loadDataSet():
	postingList = [ 
		    ['my','dog','has','flea','problems','help','please'],
		    ['maybe','not','take','him','to','dog','park','stupid'],
		    ['my','dalmation','is','so','cute','I','love','him'],
		    ['stop','posting','stupid','worthless','garbage'],
		    ['mr','licks','ate','my','steak','how','to','stop','him'],
		    ['quit','buying','worthless','dog','food','stupid']         ]
	classVec = [0,1,0,1,0,1]  #1:侮辱性文字，0:正常言论 
	return postingList,classVec

def createVocabList(dataSet):
	vocabSet = set([])        #创建一个空集 
	for document in dataSet:
		vocabSet = vocabSet | set(document)  #创建两个集合的并集，去重操作
	return list(vocabSet)  

def setOfWords2Vec(vocabList,inputSet):    #对单词做简单的词向量
	returnVec = [0] * len(vocabList) 
	for word in inputSet:
		if word in vocabList: 
			returnVec[vocabList.index(word)] = 1 
		else:
			print ("the word: %s is not in my Vocabulary!" % word)

	return returnVec

#训练函数 
def trainNB0(trainMatrix,trainCategory):
	#trainMatrix     : 0,1表示的文档矩阵
	#trainCategory   : 类别标签构成的向量

	numTrainDocs = len(trainMatrix) 
	numWords     = len(trainMatrix[0]) 
	pAbusive     = sum(trainCategory) / float(numTrainDocs)  #P(c1)
	p0Num        = np.ones(numWords) 
	p1Num        = np.ones(numWords)    #change to ones() 
	p0Denom      = 2.0                  #change to 2.0
	p1Denom      = 2.0                  #防止概率乘积为0
	#len(trainMatrix)  == len(trainCategory)
	for i in range(numTrainDocs):
		if trainCategory[i] == 1:
			p1Num    += trainMatrix[i] 
			p1Denom  += sum(trainMatrix[i])
		else:
			p0Num     += trainMatrix[i]
			p0Denom   += sum(trainMatrix[i])

	p0Vect = np.log10(p0Num / p0Denom)                     
	p1Vect = np.log10(p1Num / p1Denom)   #对向量中的逐个元素取对数，使用np.log10()而不是log()               
	return p0Vect,p1Vect,pAbusive        #取对数为了防止下溢出

 

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + log(pClass1)    #element-wise mult
    p0 = sum(vec2Classify * p0Vec) + log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else: 
        return 0

def testingNB():
	listPosts,listClasses = loadDataSet()
	myVocabList = createVocabList(listPosts)
	trainMat=[]
	for postinDoc in listPosts:
		trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
	p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
	testEntry = ['love', 'my', 'dalmation']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
	testEntry = ['stupid', 'garbage']
	thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
	print (testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))

testingNB()  


