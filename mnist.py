# -*- coding: utf-8 -*-    
'''''Trains a simple deep NN on the MNIST dataset.
'''
  
from __future__ import print_function  
  
import keras  
from keras.datasets import mnist  
from keras.models import Sequential  
from keras.layers import Dense, Dropout  
from keras.optimizers import RMSprop
from keras import backend as K
import numpy as np
  
batch_size = 128  
num_classes = 10  
epochs = 20  
  
# the data, shuffled and split between train and test sets   
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# 由于网络原因无法加载网上数据集，先下载数据集后通过本地文件载入

path='./mnist.npz'  #载入 mnisnt 数据集文件
f = np.load(path)  
x_train, y_train = f['x_train'], f['y_train']  
x_test, y_test = f['x_test'], f['y_test']  
f.close()  
  
x_train = x_train.reshape(60000, 784).astype('float32')  #返回 60000*784 的矩阵 （784=28*28）
x_test = x_test.reshape(10000, 784).astype('float32')  
x_train /= 255  #归一化
x_test /= 255  
print(x_train.shape[0], 'train samples')  #训练集数据量
print(x_test.shape[0], 'test samples')  #测试集数据量
  
# convert class vectors to binary class matrices  
# label为0~9共10个类别，keras要求格式为binary class matrices  
  
y_train = keras.utils.to_categorical(y_train, num_classes)  
y_test = keras.utils.to_categorical(y_test, num_classes)  

# Dense of keras is full-connection.  
model = Sequential()  
model.add(Dense(512, activation='relu', input_shape=(784,)))  #激活函数 relu （负值全为零 单侧抑制）
model.add(Dropout(0.2))  
model.add(Dense(512, activation='relu'))  
model.add(Dropout(0.2))  
model.add(Dense(num_classes, activation='softmax'))  
  
model.summary()  
  
model.compile(loss='categorical_crossentropy',  
              optimizer=RMSprop(),  
              metrics=['accuracy'])  
  
history = model.fit(x_train, y_train,  
                    batch_size=batch_size,  
                    epochs=epochs,  
                    verbose=1,  
                    validation_data=(x_test, y_test))  
score = model.evaluate(x_test, y_test, verbose=0)  
print('Test loss:', score[0])  
print('Test accuracy:', score[1])

K.clear_session()
#报错 AttributeError: 'NoneType' object has no attribute 'TF_NewStatus'