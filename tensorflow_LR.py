# -*- coding: utf-8 -*-
"""
Created on Tue Apr 28 14:32:00 2020
    TensorFlow学习之实现线性回归
@author: Wang
"""

import tensorflow as tf


# def linear_regression():
    # 1.准备数据
X=tf.random_normal(shape=[100,1])
y_true=tf.matmul(X,[[0.8]])+0.7

# 2.构造模型
weights=tf.Variable(initial_value=tf.random_normal(shape=[1,1]),trainable=True) # 是否可以被训练，默认为true
bias=tf.Variable(initial_value=tf.random_normal(shape=[1,1]))
y_predict=tf.matmul(X,weights)+bias

# 3.构造损失函数
error=tf.reduce_mean(tf.square(y_predict-y_true))

# 4.优化损失
optimizer=tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(error)

# 显示初始化变量
init=tf.global_variables_initializer()
sess=tf.Session()
# 开启会话
with tf.Session() as sess:
    sess.run(init)
    
    # 查看吃实话模型参数后的值
    print("训练前的模型参数为：权重%f,偏执%f,损失%f"%(weights.eval(),bias.eval(),error.eval()))
    
    for i in range(1000):
        sess.run(optimizer)
        print("堤%d次训练后的模型参数为：权重%f,偏执%f,损失%f"%(i+1,weights.eval(),bias.eval(),error.eval()))

        
# if __name__=='__main__':
#     linear_regression()



