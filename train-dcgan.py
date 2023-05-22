#!/usr/bin/env python3

# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/main.py
#   + License: MIT
# [2016-08-05] Modifications for Inpainting: Brandon Amos (http://bamos.github.io)
#   + License: MIT

import os                                                   #将os包导入
import scipy.misc                                           #导入misc 很好用的图像保存读取以及处理的包，scipy.misc
import numpy as np                                          #将numpy包导入

from model import DCGAN                                    
from utils import pp, visualize, to_json                    #将utils.py中的pp,visualize,to_json函数导入

import tensorflow as tf                                     #导入TensorFlow包

flags = tf.app.flags
#tf.app.flags用于传递tf.app.run( )所需的参数, 可查看源码flags.py ，亦可理解为处理命令行参数的解析工作 。
#查看flags.py，返回的即是FLAGS。So若调用其中的参量，形式为flags.FLAGS.XXX
flags.DEFINE_integer("epoch", 30, "Epoch to train [25]")                           #epoch次数  
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam [0.0002]")  #adam学习速率
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")                    #adam的动量
flags.DEFINE_float("train_size", np.inf, "The size of train images [np.inf]")      #训练图像的size
flags.DEFINE_integer("batch_size", 64, "The size of batch images [64]")            #批处理图像的大小  64个图像一组  可以自己修改
flags.DEFINE_integer("image_size", 64, "The size of image to use")                 #图像的大小        输入图像的大小
flags.DEFINE_string("dataset", "lfw-aligned-64", "Dataset directory.")             #数据集目录
flags.DEFINE_string("checkpoint_dir", "checkpoint", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
FLAGS = flags.FLAGS
#查看flags.py，返回的即是FLAGS。So若调用其中的参量，形式为flags.FLAGS.XXX

#要是checkpoint和samples的路径不存在,创建改路径
if not os.path.exists(FLAGS.checkpoint_dir):
    os.makedirs(FLAGS.checkpoint_dir)
if not os.path.exists(FLAGS.sample_dir):
    os.makedirs(FLAGS.sample_dir)

config = tf.ConfigProto()
#tf.ConfigProto一般用在创建session的时候，用来对session进行参数配置.
#而tf.GPUOptions可以作为设置tf.ConfigProto时的一个参数选项，一般用于限制GPU资源的使用。
config.gpu_options.allow_growth = True
#动态使用显存  当使用GPU时候，Tensorflow运行自动慢慢达到最大GPU的内存
with tf.Session(config=config) as sess:                #让参数设置生效的方法
    dcgan = DCGAN(sess, image_size=FLAGS.image_size, batch_size=FLAGS.batch_size,
                  is_crop=False, checkpoint_dir=FLAGS.checkpoint_dir)
    #DGAN为model中的一个类   
    dcgan.train(FLAGS)                 #进行训练
