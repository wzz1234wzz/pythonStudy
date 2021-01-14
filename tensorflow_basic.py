# -*- coding: utf-8 -*-
"""
Created on Sun Apr 26 11:32:27 2020

@author: Wang
"""

import tensorflow as tf
tf.disable_eager_execution()

# 以两个向量相加为例给出计算图。假设有两个向量 v_1 和 v_2 将作为输入提供给 Add 操作
v1=tf.constant([1,2,3,4]) # 定义向量常量
v2=tf.constant([2,1,5,3])
v_add= tf.add(v1,v2)

with tf.Session() as sess:
    print(sess.run(v_add))
    
print(tf.Session().run(tf.add(tf.constant([1,2,3,4]),tf.constant([2,1,5,3]))))


v3=tf.zeros([3,2],tf.int32) # 创建元素为0 的张量
print(tf.Session().run(v3))

v4=tf.zeros_like(v2) # 创建一个与v2相同尺寸全0向量
print(tf.Session().run(v4))

v5=tf.ones_like(v2)   # 创建一个与v2相同尺寸全1向量
print(tf.Session().run(v5))

# 在一定范围内生成一个从初值到终值等差排布的序列： 
# tf.linspace(start,stop,num)
range_t = tf.linspace(2.0,5.0,5)
print(tf.Session().run(range_t))

#从开始（默认值=0）生成一个数字序列，增量为 delta（默认值=1），直到终值（但不包括终值）： 
#tf.range(start,limit,delta)
range_t = tf.range(10)
print(tf.Session().run(range_t))

# 随机生成的张量受初始种子值的影响。要在多次运行或会话中获得相同的随机数，
# 应该将种子设置为一个常数值。当使用大量的随机张量时，可以使用 
# tf.set_random_seed() 来为所有随机产生的张量设置种子。以下命令将
# 所有会话的随机张量的种子设置为 54： 
tf.set_random_seed(54)

# TensorFlow 允许创建具有不同分布的随机张量：
# 使用以下语句创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的正态分布随机数组
t_random = tf.random_normal([2,3],mean=2.0,stddev=4,seed=12)
print(tf.Session().run(t_random))

# 创建一个具有一定均值（默认值=0.0）和标准差（默认值=1.0）、形状为 [M，N] 的截尾正态分布随机数组：
t_random = tf.truncated_normal([2,3],mean=2.0,stddev=4,seed=12)
print(tf.Session().run(t_random))
print('------------------------------')

# 要在种子的 [minval（default=0），maxval] 范围内创建形状为 [M，N] 的给定伽马分布随机数组，请执行如下语句：
t_random = tf.random_uniform([2,3],maxval=4,seed=12)
print(tf.Session().run(t_random))
print('------------------------------')

# 要将给定的张量随机裁剪为指定的大小，使用以下语句： 
tf.random_crop(t_random,[2,5],seed=12)
print(tf.Session().run(t_random))

print('------------------------------')
# 使用 tf.random_shuffle() 来沿着它的第一维随机排列张量。如果 t_random 是想要重新排序的张量，使用下面的代码：
v1=tf.random_shuffle(t_random)
print(tf.Session().run(v1))



# 变量的定义将指定变量如何被初始化，但是必须显式初始化所有的声明
# 变量。在计算图的定义中通过声明初始化操作对象来实现：
initial_op=tf.global_variables_initializer()

# 保存变量：使用 Saver 类来保存变量，定义一个 Saver 操作对象： 
saver = tf.train.Saver()

# 占位符，它们用于将数据提供给计算图。可以使用以下方法定义一个占位符：
# tf.placeholder(dtype,shape=None,name=None)
# dtype 定占位符的数据类型，并且必须在声明占位符时指定。在这里，
# 为 x 定义一个占位符并计算 y=2*x，使用 feed_dict 输入一个随
# 机的 4×5 矩阵
x=tf.placeholder('float')
y=2*x
data=tf.random_uniform([4,5],10)
with tf.Session() as sess:
    x_data=sess.run(data)
    print(sess.run(y,feed_dict={x:x_data}))
# 会话的run方法
# run(fetches,feed_dict=None,options=None,run_metadata=None)
#   fetches：单一的操作，或者列表、元组
#   feed_dict:参数允许使用覆盖图中张量的值，运行时赋值，与placeholder搭配

# 创建一个常量运算操作，产生一个 1×2 矩阵 
matrix1 = tf.constant([[3., 3.]]) 
# 创建另外一个常量运算操作，产生一个 2×1 矩阵 
matrix2 = tf.constant([[2.],[2.]]) 
# 创建一个矩阵乘法运算 ，把matrix1和matrix2作为输入 
# 返回值product代表矩阵乘法的结果 
product = tf.matmul(matrix1, matrix2)
print(tf.Session().run(product))


input1 = tf.placeholder(tf.float32) 
input2 = tf.placeholder(tf.float32) 
output = tf.multiply(input1, input2)  
print(tf.Session().run([output], feed_dict={input1:[7.], input2:[2.]})) 

new_g=tf.Graph() # 自定义图
with new_g.as_default():
    a_new=tf.constant(20)
    b_new=tf.constant(3)
    c_new=a_new+b_new

with tf.Session(graph=new_g) as new_sess:
    print(new_sess.run([a_new,b_new,c_new]))

# 查看图属性
print(tf.get_default_graph())# 默认图的地址
print(product.graph) # 默认图的地址
print(c_new.graph) # 新图的地址

# 改变类型
l_cast=tf.cast(output,dtype=tf.int32) # 不会改变原始
print(output) 
print(l_cast) 

# 改变形状
input21 = tf.placeholder(tf.float32,shape=[None,None]) 
input22 = tf.placeholder(tf.float32,shape=[None,3]) 
input23 = tf.placeholder(tf.float32,shape=[2,3]) 
print(input21) # 行列都可改变
print(input22) # 只能改变行
print(input23) # 已完全固定

# 静态形状set_shape：只有没有完全固定下来的形状才可以改变
input21.set_shape([3,4])
input22.set_shape([3,3])
print(input21) # 改变原始张量
print(input22) # 

# 动态形状reshape,不改变原始
l1=tf.reshape(input21,[4,3])
l2=tf.reshape(input22,[3,1,3])
print(input21) # 
print(input22) # 
print(l1) # 
print(l2) # 

#TensorFlow 变量
# 它们通过使用变量类来创建。变量的定义还包括应该初始化的常量/随机值。
# 下面的代码中创建了两个不同的张量变量 t_a 和 t_b。两者将被初始化为形
# 状为 [50，50] 的随机均匀分布，最小值=0，最大值=10：
print('------------------------------')
rand_t = tf.random_uniform([2,3],0,10,seed=12)
with tf.variable_scope("my_scope"): # 修改变量的命名空间,使得结构更加清晰
    t_a=tf.Variable(rand_t)
    t_b=tf.Variable(rand_t)
print(t_a)
print(t_b)

# 变量要显示初始化
init=tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(t_a))
    print(sess.run(t_b))

a = tf.constant([[1.0, 2.0], [1.0, 2.0], [1.0, 2.0]]) 
sess = tf.Session() 
print(sess.run(tf.sigmoid(a)))


sess = tf.Session() 
sess.run(tf.global_variables_initializer())


