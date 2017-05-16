#!/usr/bin/python
# -*- coding=utf-8 -*-
"""
MNIST DNN demo by tobycc
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
#------------------------DNN训练配置-------------------------------------
# 全链接网络结构配置
# 输入层784个节点为MNIST输入样本的特征，一层4个节点的隐藏层
# 一层4个节点的隐藏层
# 输出层SoftMax10个节点为MNIST样本的label
NET_CONFIG = [784, 4, 10]
# NET_CONFIG = [784, 500, 100, 10]
# 单次batch样本数
BATCH_SIZE = 100
# 最大训练轮数（batch）
MAX_ITER = 10000

# 隐藏层激活函数，支持tanh、sigmoid、relu、linear
ACTIVATION = 'tanh'
# 学习率
LEARNING_RATE = 0.8
#------------------------------------------------------------------------


#--------------------构造DNN训练计算图-----------------------------------
# 存储模型训练结果
model_layer_weight = []
model_layer_bias = []

# 输入训练数据Tensor的占位符，shape=[BATCH_SIZE, 特征数]
input_feature = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NET_CONFIG[0]))
# 输入真实label Tensor的占位符，shape=[BATCH_SIZE, label数]
input_label = tf.placeholder(tf.float32, shape=(BATCH_SIZE, NET_CONFIG[-1]))

# 前向传播过程
# 初始化当前层输入节点数
in_dim = NET_CONFIG[0]
# 从输入层开始，循环往下层进行前向传播
cur_layer = input_feature
for i in range(1, len(NET_CONFIG)):
    # 当前层输出接点数
    out_dim = NET_CONFIG[i]
    
    # 当前层weight Tensor，shape=[in_dim, out_dim]
    cur_layer_weight = tf.Variable(tf.random_normal([in_dim, out_dim]), dtype = tf.float32)
    # 当前层bias Tensor，shape=[out_dim]
    cur_layer_bias = tf.Variable(tf.random_normal([out_dim]), dtype = tf.float32)
    # 添加到模型存储结果
    model_layer_weight.append(cur_layer_weight)
    model_layer_bias.append(cur_layer_bias)

    # 前向传播，进行线性矩阵变换 
    # 当前层输入矩阵 X 当前层weight + 当前层bias
    cur_layer = tf.matmul(cur_layer, cur_layer_weight) + cur_layer_bias

    # 根据设置的激活函数，进行非线性变化
    if ACTIVATION == 'tanh':
        cur_layer = tf.nn.tanh(cur_layer)
    elif ACTIVATION == 'sigmoid':
        cur_layer = tf.nn.sigmoid(cur_layer)
    elif ACTIVATION == 'relu':
        cur_layer = tf.nn.relu(cur_layer)

    # 进入下一层，更新输入节点数
    in_dim = NET_CONFIG[i]

# 得到前向传播计算结果，cur_layer为batch数据的预测结果Tensor, shape=[BATCH_SIZE, label数(NET_CONFIG[-1])]
# 进行softmax变换
predict_label = tf.nn.softmax(cur_layer)

# 计算当前batch的损失函数值
# 预测结果Tensor和当前batch实际结果Tensor的交叉熵
loss = -tf.reduce_mean(input_label * tf.log(predict_label))
#loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = cur_layer, labels = tf.argmax(input_label, 1)))

# 后向传播过程
# 计算每个节点相对损失函数的梯度
# 根据后向传播结果，对model_layer_weight和model_layer_bias进行梯度下降，更新权重
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
#------------------------------------------------------------------------


#--------------------执行DNN训练计算图-----------------------------------
# 加载MNIST数据集，包括train、validation、test三部分
mnist_dataset = input_data.read_data_sets("./MNIST_data", one_hot=True)
with tf.Session() as sess:
    # 初始化计算图中的所有Variables
    sess.run(tf.global_variables_initializer())
    
    for i in range(MAX_ITER):
        # 读取一个batch的训练数据
        train_batch_feature, train_batch_label = mnist_dataset.train.next_batch(BATCH_SIZE)
        # 执行DNN训练计算图，完成在当前batch数据上的一次前向传播+后向传播过程
        # 通过Feed机制指定当前batch数据作为输入Tensor，通过Fetch机制获取当前损失函数值
        _train, loss_value = sess.run([train_step, loss], 
                feed_dict={input_feature: train_batch_feature, input_label: train_batch_label})
        # 每过1000个batch输出在当前训练batch上的损失函数值
        if i % 1000 == 0:
            print '''After %d training step(s), loss on training batch is %g.''' % (i, loss_value)

    # 模型训练结束，输出模型结果
    print '''Finish Model Train:'''
    for i in range(0, len(model_layer_weight)):
        print '''Layer-Weight-%d:''' % i
        print model_layer_weight[i].eval()
        print '''Layer-Bias-%d:''' % i
        print model_layer_bias[i].eval()
    print '''MNIST DNN All Finish'''
#------------------------------------------------------------------------

