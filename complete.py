#!/usr/bin/env python3
#
# Brandon Amos (http://bamos.github.io)
# License: MIT
# 2016-08-05

import argparse
import os
import tensorflow as tf

from model import DCGAN

parser = argparse.ArgumentParser()
parser.add_argument('--approach', type=str,
                    choices=['adam', 'hmc'],
                    default='adam')
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--eps', type=float, default=1e-8)
parser.add_argument('--hmcBeta', type=float, default=0.2)
parser.add_argument('--hmcEps', type=float, default=0.001)
parser.add_argument('--hmcL', type=int, default=100)
parser.add_argument('--hmcAnneal', type=float, default=1)
parser.add_argument('--nIter', type=int, default=1000)                 #迭代次数
parser.add_argument('--imgSize', type=int, default=64)                 #默认图片大小   可以自己修改
parser.add_argument('--lam', type=float, default=0.1)      
parser.add_argument('--checkpointDir', type=str, default='checkpoint')           #训练模型的保存
parser.add_argument('--outDir', type=str, default='completions')                 #保存的类型
parser.add_argument('--outInterval', type=int, default=50)                       #每多少次输出一次
parser.add_argument('--maskType', type=str,                                      #遮盖的类型
                    choices=['random', 'center', 'left', 'full', 'grid', 'lowres'],
                    default='center')
parser.add_argument('--centerScale', type=float, default=0.25)                    #中心的大小

parser.add_argument('imgs', type=str, nargs='+' )  #自己改的                      #修补图片的路径
#parser.add_argument('imgs', type=str, default = ' ./samples1/903_faceimage67418.jpg   ' )  #自己改的
args = parser.parse_args()   

assert(os.path.exists(args.checkpointDir))

#imgsset = r"C:\Users\hasee\.spyder-py3\dcgan-completion.tensorflow-master\samples1\*.tiff"  #自己改的
#imgs=args.imgset
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    dcgan = DCGAN(sess, image_size=args.imgSize,
                  batch_size=min(64, len(args.imgs)),
                  checkpoint_dir=args.checkpointDir, lam=args.lam)
    #dcgan = DCGAN(sess, image_size=args.imgSize,
                  #batch_size=min(64, len(imgs)),
                  #checkpoint_dir=args.checkpointDir, lam=args.lam)    #自己改的
    dcgan.complete(args)