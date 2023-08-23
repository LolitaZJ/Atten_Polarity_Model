#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 11:31:28 2023

@author: zhangj2
"""


# In[] Libs
import os
import numpy as np
os.getcwd()
import argparse

import datetime
import keras
from keras import losses
from keras import optimizers
from keras.models import Sequential
from keras.models import Model,load_model
from keras.layers import Input, Dense, Dropout, Flatten,Embedding, LSTM,GRU,Bidirectional
from keras.layers import Conv1D,Conv2D,MaxPooling1D,MaxPooling2D,BatchNormalization,Reshape
from keras.layers import UpSampling1D,AveragePooling1D,AveragePooling2D,TimeDistributed 
from keras.layers import UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers import Lambda,concatenate,add,Conv2DTranspose,Concatenate
from keras.callbacks import LearningRateScheduler,EarlyStopping,ModelCheckpoint
from keras.layers import Reshape

from keras_self_attention import SeqSelfAttention
from keras.utils.np_utils import to_categorical
from keras import backend as K
import tensorflow as tf


# In[]  A size that is an integer multiple of 8
def build_pd_model(time_input=(400,1),num_dense=128,clas=3):
    
    inp = Input(shape=time_input, name='input')
    
    x = Conv1D(16, 5, padding = 'same', activation = 'relu')(inp)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(32, 3, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)    
    
    x = Conv1D(64, 3, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = keras.layers.LSTM(units=128, return_sequences=True)(x)
    
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= 20,  
                            attention_activation='relu',name='Atten')(x)

    #----------------------#
    x = Flatten()(at_x)
    
    x = Dense(num_dense,activation = 'relu')(x)
    
    out2 = Dense(clas,activation = 'softmax',name='po')(x)
    
    model = Model(inp, out2)
    
    return model
# In[]
def build_ross_model(time_input=(400,1),num_dense=512,clas=3):
    
    inp = Input(shape=time_input, name='input')
    
    x = Conv1D(32, 21, padding = 'same', activation = 'relu')(inp)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(64, 15, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)    
    x = MaxPooling1D(2)(x)
    
    x = Conv1D(128, 11, padding = 'same', activation = 'relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    
    #----------------------#
    x = Flatten()(x)
    x = Dense(num_dense,activation = 'relu')(x)
    x = BatchNormalization()(x)
    
    x = Dense(num_dense,activation = 'relu')(x)
    x = BatchNormalization()(x)  
    
    out2 = Dense(clas,activation = 'softmax',name='po')(x)
    model = Model(inp, out2)
    return model

# In[]
def build_PP_model(time_input=(400,1),clas=3,filter_size=3,num_filter=[16,32,64],num_dense=128):
    
    inp = Input(shape=time_input, name='input')
    # print(num_filter)
    x = Conv1D(num_filter[0], filter_size+2, padding = 'same', activation = 'relu')(inp)
    
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(num_filter[1], filter_size, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = Conv1D(num_filter[2], filter_size, padding = 'same', activation = 'relu')(x)
    x = MaxPooling1D(2)(x)
    x = BatchNormalization()(x)
    
    x = keras.layers.LSTM(units=num_filter[2]*2, return_sequences=True)(x)
    
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= 20,  
                            attention_activation='relu',name='Atten')(x)
    #----------------------#
    x1 = UpSampling1D(2)(at_x)
    x1 = Conv1D(num_filter[2], filter_size, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = UpSampling1D(2)(x1)
    x1 = Conv1D(num_filter[1], filter_size, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    x1 = UpSampling1D(2)(x1)
    x1 = Conv1D(num_filter[0], filter_size, padding = 'same', activation = 'relu')(x1)
    x1 = BatchNormalization()(x1)
    
    out1 = Conv1D(1, filter_size, padding = 'same', activation = 'sigmoid',name='pk')(x1)
    
    #----------------------#
    x = Flatten()(at_x)
    
    x = Dense(num_dense,activation = 'relu')(x)
    
    out2 = Dense(clas,activation = 'softmax',name='po')(x)
    
    model = Model(inp, [out1,out2])
    
    return model



# In[] suit for any input

def Conv2d_BN1(x, nb_filter, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2d_BN2(x, nb_filter, kernel_size, strides=(4,1), padding='same'):
    x = Conv2D(nb_filter, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN1(x, filters, kernel_size, strides=(4,1), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN2(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2DTranspose(filters, kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x
    
def Conv2dT_BN3(x, filters, kernel_size, strides=(1,1), padding='same'):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
    x = UpSampling2D(size=(4,1))(x) #1
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    return x    
    
def crop_and_cut(net):
    net1,net2=net
    net1_shape = net1.get_shape().as_list()
    # net2_shape = net2.get_shape().as_list()
    offsets = [0, 0, 0, 0]
    size = [-1, net1_shape[1], net1_shape[2], -1]
    net2_resize = tf.slice(net2, offsets, size)
    return net2_resize 

# def my_reshape(x,a,b):
#     return K.reshape(x,(-1,a,b)) 

# x2=Lambda(my_reshape,arguments={'a':750*2,'b':4*3})(inpt)

    
def pd_model(time_input,num=1,nb_filter=8, kernel_size=(7,1),depths=5,clas=3,num_dense=128):
    
    inpt = Input(shape=time_input,name='input')
    # Down/Encode
    convs=[None]*depths
    net = Conv2d_BN1(inpt, nb_filter, kernel_size)
    for depth in range(depths):
        filters=int(2**depth*nb_filter)
        
        net = Conv2d_BN1(net, filters, kernel_size)
        convs[depth] = net
    
        if depth < depths - 1:
            net = Conv2d_BN2(net, filters, kernel_size)
    # Reshape        
    net_shape = net.get_shape().as_list()       
    net=Reshape((net_shape[1],net_shape[3]))(net)
    # LSTM
    net = keras.layers.LSTM(units=filters, return_sequences=True)(net)
    # Attention
    at_x,wt = SeqSelfAttention(return_attention=True, attention_width= 20,  
                            attention_activation='relu',name='Atten')(net)   
    x = Flatten()(at_x)
             
    #=====================#
    x = Dense(num_dense,activation = 'relu')(x)
    
    outenv = Dense(clas,activation = 'softmax',name='po')(x)
    
    model = Model(inpt, [outenv],name='pd_model')
    return model   