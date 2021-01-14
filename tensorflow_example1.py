# -*- coding: utf-8 -*-
"""
    图像识别
"""
import tensorflow as tf
import os

filename=os.listdir('./dog') # 获取如片的文件名
# 将文件名与路径进行拼接
file_list=[os.path.join("./dog/",file) for file in filename]

# 1.构造文件名队列
file_queue=tf.train.string_input_producer(file_list)

# 2.读取与解码
# 读取阶段
reader=tf.WholeFileReader()
key,value=reader.read(file_queue) # 返回的是Tensor对象
print("文件名key：",key)
print("文件内容value：",value)

# 解码阶段
image=tf.image.decode_jpeg(value)
print("image：",image)

# 图像的类型修改
image_resize=tf.image.resize_images(image,[200,200])
print("image_resize：",image_resize)

# 静态形状改变
image_resize.set_shape(shape=[200,200,3])

# 3.批处理
image_batch=tf.train.batch([image_resize],batch_size=5,num_threads=1,capacity=5)
print("image_batch：",image_batch)

# 开启会话
with tf.Session() as sess:
    coord=tf.train.Coordinator()  # 线程协调员
    thread=tf.train.start_queue_runners(sess=sess,coord=coord) # 开启线程
    new_key,new_value,new_image,new_image_resize,new_image_batch=sess.run([key,value,image,image_resize,image_batch])
    print("文件名：",new_key)
    print("文件内容：",new_value)
    print("new_image：",new_image)
    print("new_image_resize：",new_image_resize)
    print("new_image_batch：",new_image_batch)

    coord.request_stop() # 请求暂停
    coord.join(thread) # 回收线程

















