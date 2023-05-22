# Original Version: Taehoon Kim (http://carpedm20.github.io)
#   + Source: https://github.com/carpedm20/DCGAN-tensorflow/blob/e30539fb5e20d5a0fed40935853da97e9e55eee8/model.py
#   + License: MIT
# [2016-08-05] Modifications for Completion: Brandon Amos (http://bamos.github.io)
#   + License: MIT

from __future__ import division                                    #支持使用精确除法
import os                                                          #导入os包                                                        
import time                                                        #导入time包
import math                                                        #导入math包
import itertools                                                   #导入itertools包
                                                                   #itertools为我们提供了非常有用的用于操作迭代对象的函数
from glob import glob                                              #该包包含两个主要的函数glob和iglob
                                                                   #输入参数均是文件路径，返回值glob为文件列表（无序），iglob为迭代器
import tensorflow as tf                                            #导入tensorflow包
from six.moves import xrange
#Six提供了简单的实用程序包来封装Python 2和Python 3之间的差异。它旨在支持无需修改即可在Python 2和Python 3上工作的代码库。
#用xrange会比range性能优很多，因为不需要一上来就开辟一块很大的内存空间。


from ops import *                                                   #导入ops.py中的所有方法
from utils import *                                                 #导入utils.py中的所有方法

SUPPORTED_EXTENSIONS = ["png", "jpg", "jpeg","tiff"]                #支持的图片格式

dataset = r"F:\360Downloads\faces_64"  #自己改的

epoch=30     #自己改的

#imgs = r"C:\Users\hasee\.spyder-py3\dcgan-completion.tensorflow-master\samples1\*.tiff"  #自己改的

#outDir = r"C:\Users\hasee\.spyder-py3\dcgan-completion.tensorflow-master\outputImages"

#imgs=args.imgset

#dataset_files函数  返回给定目录中所有图像文件的列表  
#输入是文件夹目录
#输出是  list 
#itertools.chain 连接多个列表或者迭代器
#若iterables 事先不能确定，可以使用chain.from_iterable()函数   
#itertools中函数设计的初衷是使用起来快速且更有效的利用内存，数据不会被创建直到真的需要，这种“lazy”模式使其不用存储大量数据在内存中。

def dataset_files(root):
    """Returns a list of all image files in the given directory"""
    return list(itertools.chain.from_iterable(
        glob(os.path.join(root, "*.{}".format(ext))) for ext in SUPPORTED_EXTENSIONS))               
#用法os.path.join(path1, path2, ...)返回：多个路径拼接组合
#glob匹配的文件路径列表，list类型   注：只能遍历当前文件夹下的文件，不能遍历子文件夹中的文件


#核心   DCGAN类
class DCGAN(object):

	#  __init__ 函数    类起到模板的作用，因此，可以在创建实例的时候，把我们认为必须绑定的属性强制填写进去。这里就用到Python当中的一个内置方法__init__方法，
	#这样一来，我们从外部看Student类，就只需要知道，创建实例需要给出name和score。而如何打印，都是在Student类的内部定义的，这些数据和逻辑被封装起来了，调用很容易，但却不知道内部实现的细节。
	#另外，这里self就是指类本身，self.name就是Student类的属性变量，是Student类所有。而name是外部传来的参数，不是Student类所自带的。故，self.name = name的意思就是把外部传来的参数name的值赋值给Student类自己的属性变量self.name。
	#既然Student类实例本身就拥有这些数据，那么要访问这些数据，就没必要从外面的函数去访问，而可以直接在Student类的内部定义访问数据的函数（方法），这样，就可以把”数据”封装起来。这些封装数据的函数是和Student类本身是关联起来的，
    def __init__(self, sess, image_size=64, is_crop=False,
                 batch_size=64, sample_size=64, lowres=8,
                 z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3,
                 checkpoint_dir=None, lam=0.1):
    #需要注意的是，在Python中，变量名类似__xxx__的，也就是以双下划线开头，并且以双下划线结尾的，是特殊变量，特殊变量是可以直接访问的，不是private变量，所以，不能用__name__、__score__这样的变量名。
        """
        Args:
            sess: TensorFlow session                                                         #sess:TensorFlow回话
            batch_size: The size of batch. Should be specified before training.              #batch_size:批处理的大小。应在培训前指定。
            lowres: (optional) Low resolution image/mask shrink factor. [8]                  #低分辨率图像/掩模收缩因子。
            z_dim: (optional) Dimension of dim for Z. [100]                                  #Z的dim尺寸
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]            #在第一conv层的gen过滤器的尺寸。
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]        #描述滤波器在第一conv层的尺寸。
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024] #全连通层的gen untis尺寸
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024] #全连通层的描述单元尺寸。
            c_dim: (optional) Dimension of image color. [3]                                  #Dimension of image color.
        """
        # Currently, image size must be a (power of 2) and (8 or higher).
        #
        #如果图片格式出错则报错
        assert(image_size & (image_size - 1) == 0 and image_size >= 8)    
        #限制图片格式

        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = [image_size, image_size, c_dim]

        self.lowres = lowres
        self.lowres_size = image_size // lowres
        self.lowres_shape = [self.lowres_size, self.lowres_size, c_dim]

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.lam = lam

        self.c_dim = c_dim
        #用self给DCGAN的属性赋值

        # batch normalization : deals with poor initialization helps gradient flow
        #批处理规范化:处理糟糕的初始化有助于梯度流

        #大概就是初始化 没有怎么懂 

        self.d_bns = [
            #batch_norm(name='d_bn{}'.format(i,)) for i in range(4)]
            #ops.py中的类
            batch_norm(name='d_bn{}'.format(i,)) for i in range(1,4)]
        log_size = int(math.log(image_size) / math.log(2))
        self.g_bns = [
            #batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size)]             #自己改的   用于解决读取模型出错
            batch_norm(name='g_bn{}'.format(i,)) for i in range(log_size - 1)]

        self.checkpoint_dir = checkpoint_dir     #读取以前的训练模型路径
        self.build_model()                       #初始化的model 

        self.model_name = "DCGAN.model"			#命名
    #build_model函数   利用tf类里的一些工具来初始化model
    def build_model(self):
    	#tf.placeholder(dtype, shape=None, name=None)
		#此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值		#参数：
		#dtype：数据类型。常用的是tf.float32,tf.float64等数值类型
		#shape：数据形状。默认是None，就是一维值，也可以是多维，比如[2,3], [None, 3]表示列是3，行不定
		#name：名称。

        self.is_training = tf.placeholder(tf.bool, name='is_training')
        self.images = tf.placeholder(
            tf.float32, [None] + self.image_shape, name='real_images')
        #定义两个形参 is_training (bool类型)   images (float32类型)   
        self.lowres_images = tf.reduce_mean(tf.reshape(self.images,
            [self.batch_size, self.lowres_size, self.lowres,
             self.lowres_size, self.lowres, self.c_dim]), [2, 4])
        #tf.reduce_mean()用于求某一维上的平均值  
        #源码：x_image = tf.reshape(x, [-1, 28, 28, 1])这里是将一组图像矩阵x重建为新的矩阵，该新矩阵的维数为（a，28，28，1），其中-1表示a由实际情况来定。 
        #例如，x是一组图像的矩阵（假设是50张，大小为56×56），则执行x_image = tf.reshape(x, [-1, 28, 28, 1])可以计算a=50×56×56/28/28/1=200。即x_image的维数为（200，28，28，1）。
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')   
        self.z_sum = tf.summary.histogram("z", self.z)
        #在训练神经网络时，当需要查看一个张量在训练过程中值的分布情况时，可通过tf.summary.histogram()将其分布情况以直方图的形式在TensorBoard直方图仪表板上显示．

        self.G = self.generator(self.z)                   				#G为 z 调用generrator生成的结果
        self.lowres_G = tf.reduce_mean(tf.reshape(self.G,
            [self.batch_size, self.lowres_size, self.lowres,
             self.lowres_size, self.lowres, self.c_dim]), [2, 4])         #求平均值
        self.D, self.D_logits = self.discriminator(self.images)           #D为调用discriminator 识别 images的结果  

        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)     #D_为调用discriminator 识别 images的结果 
        

        self.d_sum = tf.summary.histogram("d", self.D)              #直方图显示D D_
        self.d__sum = tf.summary.histogram("d_", self.D_)                          
        self.G_sum = tf.summary.image("G", self.G)
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits,
                                                    labels=tf.ones_like(self.D)))
        #为了方便查看图像预处理的效果，可以利用 TensorFlow 提供的 tensorboard 工具进行可视化。
        #使用方法也比较简单，直接用 tf.summary.image 将图像写入 summary，
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_,
                                                    labels=tf.ones_like(self.D_)))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)  
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)
        #输出一个含有标量值的Summary protocol buffer，这是一种能够被tensorboard模块解析的【结构化数据格式】 用来显示标量信息 用来可视化标量信息
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver(max_to_keep=1)

        # Completion.
        #用于图像补齐的参数部分声明
        self.mask = tf.placeholder(tf.float32, self.image_shape, name='mask')
        self.lowres_mask = tf.placeholder(tf.float32, self.lowres_shape, name='lowres_mask')
        self.contextual_loss = tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.mask, self.G) - tf.multiply(self.mask, self.images))), 1)
        self.contextual_loss += tf.reduce_sum(
            tf.contrib.layers.flatten(
                tf.abs(tf.multiply(self.lowres_mask, self.lowres_G) - tf.multiply(self.lowres_mask, self.lowres_images))), 1)
        self.perceptual_loss = self.g_loss
        self.complete_loss = self.contextual_loss + self.lam*self.perceptual_loss
        self.grad_complete_loss = tf.gradients(self.complete_loss, self.z)
    #用于训练的函数
    def train(self, config):
        data = dataset_files(dataset)      #修改过config
        #读入数据 并且打乱数据 
        np.random.shuffle(data)                                        
        assert(len(data) > 0)                      
        #要是没有读入则警告
        #设置生成器与判别器的 学习率 与 beta1
        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)            
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.g_sum = tf.summary.merge(
            [self.z_sum, self.d__sum, self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge(
            [self.z_sum, self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)
        #生成日志文件
        sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))     
        sample_files = data[0:self.sample_size]

        sample = [get_image(sample_file, self.image_size, is_crop=self.is_crop) for sample_file in sample_files]
        sample_images = np.array(sample).astype(np.float32)

        counter = 1
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("""

======
An existing model was found in the checkpoint directory.
If you just cloned this repository, it's a model for faces
trained on the CelebA dataset for 20 epochs.
If you want to train a new model from scratch,
delete the checkpoint directory or specify a different
--checkpoint_dir argument.
======

"""
#在检查点目录中找到了一个现有的模型。
#如果您刚刚克隆了这个存储库，它是一个在CelebA数据集中训练了20个代数的人脸模型。
#如果你想从头开始训练一个新模型，删除检查点目录或指定其他目录
#--checkpoint_dir argument.
)
        else:
            print("""

======
An existing model was not found in the checkpoint directory.
Initializing a new one.
======

"""
#检查点目录中没有找到现有模型。
#初始化一个新的。
)
        epoch=70            #调整迭代次数
        for epoch in xrange(epoch):
            data = dataset_files(dataset)         #修改过config
            batch_idxs = min(len(data), config.train_size) // self.batch_size

            for idx in xrange(0, batch_idxs):
                batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)
                         for batch_file in batch_files]
                batch_images = np.array(batch).astype(np.float32)

                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                            .astype(np.float32)

                # Update D network
                #更新判别网络
                _, summary_str = self.sess.run([d_optim, self.d_sum],
                    feed_dict={ self.images: batch_images, self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                # Update G network
                #更新生成网络
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                # 运行g_optim两次以确保d_loss不为零(与paper不同)
                _, summary_str = self.sess.run([g_optim, self.g_sum],
                    feed_dict={ self.z: batch_z, self.is_training: True })
                self.writer.add_summary(summary_str, counter)

                errD_fake = self.d_loss_fake.eval({self.z: batch_z, self.is_training: False})
                errD_real = self.d_loss_real.eval({self.images: batch_images, self.is_training: False})
                errG = self.g_loss.eval({self.z: batch_z, self.is_training: False})

                counter += 1
                print("Epoch: [{:2d}] [{:4d}/{:4d}] time: {:4.4f}, d_loss: {:.8f}, g_loss: {:.8f}".format(
                    epoch, idx, batch_idxs, time.time() - start_time, errD_fake+errD_real, errG))
                #循环输出  次数   时间   生成器和判别器的损失率  
                if np.mod(counter, 10) == 1:                                #100改为10了   每10次保存结果 
                    samples, d_loss, g_loss = self.sess.run(
                        [self.G, self.d_loss, self.g_loss],
                        feed_dict={self.z: sample_z, self.images: sample_images, self.is_training: False}
                    )
                    #save_images(samples, [8, 8],
                                #'./samples/train_{:02d}_{:04d}.png'.format(epoch, idx)) 
                    save_images(samples, [8, 8],
                                r"F:\samples\train_{:02d}_{:04d}.png".format(epoch, idx)) 
                    print("[Sample] d_loss: {:.8f}, g_loss: {:.8f}".format(d_loss, g_loss))
                    #保存结果

                if np.mod(counter, 500) == 2:
                    self.save(config.checkpoint_dir, counter)
                        # 当每训练 500次时  保存模型
    #用于补全的函数
    def complete(self, config):
        #产生输出路径的函数
        def make_dir(name):
            # Works on python 2.7, where exist_ok arg to makedirs isn't available.
            #在python 2.7上工作，其中exist_ok arg to makedirs是不可用的
            p = os.path.join(config.outDir, name)
            if not os.path.exists(p):
                os.makedirs(p)
        make_dir('hats_imgs')
        make_dir('completed')
        make_dir('logs')
        #创建子文件夹
        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()
        #读入训练好的模型
        isLoaded = self.load(self.checkpoint_dir)
        assert(isLoaded)
        #要是读入失败则预警
        #imgs=dataset_files(imgsset)        # 自己添加的
        nImgs = len(config.imgs)         #自己改过的 config去掉
        #nImgs = len(imgs)
        batch_idxs = int(np.ceil(nImgs/self.batch_size))  
        lowres_mask = np.zeros(self.lowres_shape)
        if config.maskType == 'random':                                           #加mask类型的   随机类型
            fraction_masked = 0.2
            mask = np.ones(self.image_shape)
            mask[np.random.random(self.image_shape[:2]) < fraction_masked] = 0.0
        elif config.maskType == 'center':                                          #中间类型
            assert(config.centerScale <= 0.5)
            mask = np.ones(self.image_shape)
            sz = self.image_size
            l = int(self.image_size*config.centerScale)
            u = int(self.image_size*(1.0-config.centerScale))
            mask[l:u, l:u, :] = 0.0
        elif config.maskType == 'left':                                              #左边类型
            mask = np.ones(self.image_shape)
            c = self.image_size // 2
            mask[:,:c,:] = 0.0
        elif config.maskType == 'full':                                                #全部遮盖
            mask = np.ones(self.image_shape)
        elif config.maskType == 'grid':                                                  #灰度
            mask = np.zeros(self.image_shape)
            mask[::4,::4,:] = 1.0
        elif config.maskType == 'lowres':                                                #下边类型
            lowres_mask = np.ones(self.lowres_shape)
            mask = np.zeros(self.image_shape)
        else:
            assert(False)

        for idx in xrange(0, batch_idxs):
            l = idx*self.batch_size
            u = min((idx+1)*self.batch_size, nImgs)
            batchSz = u-l
            batch_files = config.imgs[l:u]             #自己改的
            #batch_files = imgs[l:u]
            batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop)    #读取图片的
                     for batch_file in batch_files]
            batch_images = np.array(batch).astype(np.float32)
            if batchSz < self.batch_size:
                print(batchSz)
                padSz = ((0, int(self.batch_size-batchSz)), (0,0), (0,0), (0,0))
                batch_images = np.pad(batch_images, padSz, 'constant')
                batch_images = batch_images.astype(np.float32)

            zhats = np.random.uniform(-1, 1, size=(self.batch_size, self.z_dim))
            m = 0
            v = 0

            nRows = np.ceil(batchSz/8)
            nCols = min(8, batchSz)
            save_images(batch_images[:batchSz,:,:,:], [nRows,nCols],                 #保存原始图片
                        os.path.join(config.outDir, 'before.png'))
            masked_images = np.multiply(batch_images, mask)                          #生成遮盖后的图片
            save_images(masked_images[:batchSz,:,:,:], [nRows,nCols],                #保存遮盖后的图片
                        os.path.join(config.outDir, 'masked.png'))
            if lowres_mask.any():                                                   #添加lowers类型的mask才需要使用的
                lowres_images = np.reshape(batch_images, [self.batch_size, self.lowres_size, self.lowres,
                    self.lowres_size, self.lowres, self.c_dim]).mean(4).mean(2)
                lowres_images = np.multiply(lowres_images, lowres_mask)
                lowres_images = np.repeat(np.repeat(lowres_images, self.lowres, 1), self.lowres, 2)
                save_images(lowres_images[:batchSz,:,:,:], [nRows,nCols],
                            os.path.join(config.outDir, 'lowres.png'))
            for img in range(batchSz):                                              #  
                with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'a') as f:
                    f.write('iter loss ' +
                            ' '.join(['z{}'.format(zi) for zi in range(self.z_dim)]) +
                            '\n')

            for i in xrange(config.nIter):                              #进行训练
                fd = {
                    self.z: zhats,
                    self.mask: mask,
                    self.lowres_mask: lowres_mask,
                    self.images: batch_images,
                    self.is_training: False
                }
                run = [self.complete_loss, self.grad_complete_loss, self.G, self.lowres_G]
                loss, g, G_imgs, lowres_G_imgs = self.sess.run(run, feed_dict=fd)

                for img in range(batchSz):
                    with open(os.path.join(config.outDir, 'logs/hats_{:02d}.log'.format(img)), 'ab') as f:
                        f.write('{} {} '.format(i, loss[img]).encode())
                        np.savetxt(f, zhats[img:img+1])

                if i % config.outInterval == 0:
                    print(i, np.mean(loss[0:batchSz]))              #每50次循环输出  第一个的次数 第二个的loss
                    imgName = os.path.join(config.outDir,
                                           'hats_imgs/{:04d}.png'.format(i))
                    nRows = np.ceil(batchSz/8)
                    nCols = min(8, batchSz)
                    save_images(G_imgs[:batchSz,:,:,:], [nRows,nCols], imgName)
                    if lowres_mask.any():
                        imgName = imgName[:-4] + '.lowres.png'
                        save_images(np.repeat(np.repeat(lowres_G_imgs[:batchSz,:,:,:],
                                              self.lowres, 1), self.lowres, 2),
                                    [nRows,nCols], imgName)

                    inv_masked_hat_images = np.multiply(G_imgs, 1.0-mask)
                    completed = masked_images + inv_masked_hat_images
                    imgName = os.path.join(config.outDir,
                                           'completed/{:04d}.png'.format(i))
                    save_images(completed[:batchSz,:,:,:], [nRows,nCols], imgName)

                if config.approach == 'adam':
                    # Optimize single completion with Adam
                    #使用Adam优化单个完成
                    m_prev = np.copy(m)
                    v_prev = np.copy(v)
                    m = config.beta1 * m_prev + (1 - config.beta1) * g[0]
                    v = config.beta2 * v_prev + (1 - config.beta2) * np.multiply(g[0], g[0])
                    m_hat = m / (1 - config.beta1 ** (i + 1))
                    v_hat = v / (1 - config.beta2 ** (i + 1))
                    zhats += - np.true_divide(config.lr * m_hat, (np.sqrt(v_hat) + config.eps))
                    zhats = np.clip(zhats, -1, 1)

                elif config.approach == 'hmc':
                    # Sample example completions with HMC (not in paper)
                    # 使用HMC完成示例
                    zhats_old = np.copy(zhats)
                    loss_old = np.copy(loss)
                    v = np.random.randn(self.batch_size, self.z_dim)
                    v_old = np.copy(v)

                    for steps in range(config.hmcL):
                        v -= config.hmcEps/2 * config.hmcBeta * g[0]
                        zhats += config.hmcEps * v
                        np.copyto(zhats, np.clip(zhats, -1, 1))
                        loss, g, _, _ = self.sess.run(run, feed_dict=fd)
                        v -= config.hmcEps/2 * config.hmcBeta * g[0]

                    for img in range(batchSz):
                        logprob_old = config.hmcBeta * loss_old[img] + np.sum(v_old[img]**2)/2
                        logprob = config.hmcBeta * loss[img] + np.sum(v[img]**2)/2
                        accept = np.exp(logprob_old - logprob)
                        if accept < 1 and np.random.uniform() > accept:
                            np.copyto(zhats[img], zhats_old[img])

                    config.hmcBeta *= config.hmcAnneal

                else:
                    assert(False)
     #定义判别器
    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()

            # TODO: Investigate how to parameterise discriminator based off image size.
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bns[0](conv2d(h0, self.df_dim*2, name='d_h1_conv'), self.is_training))
            h2 = lrelu(self.d_bns[1](conv2d(h1, self.df_dim*4, name='d_h2_conv'), self.is_training))
            h3 = lrelu(self.d_bns[2](conv2d(h2, self.df_dim*8, name='d_h3_conv'), self.is_training))
            #h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h4_lin')
            h4 = linear(tf.reshape(h3, [-1, 8192]), 1, 'd_h3_lin')
            return tf.nn.sigmoid(h4), h4
     #定义生成器
    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*8*4*4, 'g_h0_lin', with_w=True)
    
            # TODO: Nicer iteration pattern here. #readability
            hs = [None]
            hs[0] = tf.reshape(self.z_, [-1, 4, 4, self.gf_dim * 8])
            hs[0] = tf.nn.relu(self.g_bns[0](hs[0], self.is_training))

            i = 1 # Iteration number.
            #depth_mul = 8  # Depth decreases as spatial component increases.
            depth_mul = 4
            size = 8  # Size increases as depth decreases.

            while size < self.image_size:
                hs.append(None)
                name = 'g_h{}'.format(i)
                hs[i], _, _ = conv2d_transpose(hs[i-1],
                    [self.batch_size, size, size, self.gf_dim*depth_mul], name=name, with_w=True)
                hs[i] = tf.nn.relu(self.g_bns[i](hs[i], self.is_training))

                i += 1
                depth_mul //= 2
                size *= 2

            hs.append(None)
            name = 'g_h{}'.format(i)
            hs[i], _, _ = conv2d_transpose(hs[i - 1],
                [self.batch_size, size, size, 3], name=name, with_w=True)
    
            return tf.nn.tanh(hs[i])
    #保存模型的函数
    def save(self, checkpoint_dir, step):
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, self.model_name),
                        global_step=step)
     #读取模型的函数
    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            return True
        else:
            return False
