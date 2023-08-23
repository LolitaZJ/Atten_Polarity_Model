#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 14:21:19 2023
@author: zhangj2

Only achieve polatiry determination based on Attention Mechanism


Quickly Start
====================================================================

python APP_PD_MASTER.py --save_name=pd_model_ins  --dataset=ins --GPU=3

python APP_PD_MASTER.py --save_name=pd_model_hinet --dataset=hinet --GPU=2

python APP_PD_MASTER.py --save_name=pd_model_scsn --dataset=scsn --GPU=1

python APP_PD_MASTER.py --save_name=pd_model_pnw --dataset=pnw -GPU=0


Parameter
====================================================================
# model_name: bulid_pd_model/cnn_ross/bulid_pp_model/pd_model

bulid_pd_model: attention-based polarity determination
cnn_ross: CNN model based Ross et al paper
bulid_pp_model: attention-based phase picking & polarity determination
pd_model: attention-based polarity determination (2D)

# save_name: the model name (pd_model_ins)

# dataset: Four dataset ins/hinet/scsn/pnw
ins: https://www.pi.ingv.it/banche-dati/instance/
hinet: https://www.hinet.bosai.go.jp/
pnw: https://github.com/niyiyu/PNW-ML/tree/main
scsn: https://scedc.caltech.edu/data/deeplearning.html#picking_polarity

# epochs: number of epochs (default: 500)

# batch_size: default=1024
                        ,
# learning_rate: default=0.001,

# patience: early stopping  (default=20)

# monitor: monitor the val_loss/loss/acc/val_acc  (default="val_accuracy")

# monitor_mode: min/max/auto (default="max")

# loss: loss fucntion  (default='categorical_crossentropy')

        

"""
# In[] Libs
import os
import numpy as np
os.getcwd()
import argparse
import matplotlib
# matplotlib.use('agg')

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

from APP_PD_model import build_pd_model, pd_model,build_ross_model,build_PP_model
from APP_PD_utils import DataGenerator_PD,DataGenerator_PD_Ins,DataGenerator_PD_Hinet,DataGenerator_PD_Scsn,plot_loss
from APP_PD_utils import confusion_matrix,DataGenerator_PD_PNW,DataGenerator_PP_PNW,plot_loss_pp,DataGenerator_PP_Scsn
import pandas as pd 
import h5py
import math
import matplotlib.pyplot as plt
plt.rc('font', family='Times New Roman')

# In[]
# Set GPU
#==========================================# 
def start_gpu(args):
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    print('Physical GPU：', len(gpus))
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print('Logical GPU：', len(logical_gpus))

#==========================================#
# Set Configures
#==========================================# 
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GPU",
                        default="3",
                        help="set gpu ids") 
    
    parser.add_argument("--model_name",
                        default="bulid_pd_model",
                        help="bulid_pd_model/cnn_ross/bulid_pp_model/pd_model")
    
    parser.add_argument("--save_name",
                        default="pd_model_ins",
                        help="model name")
    
    parser.add_argument("--dataset",
                        default="scsn",
                        help="ins/hinet/scsn/pnw")     
    
    parser.add_argument("--epochs",
                        default=500,
                        type=int,
                        help="number of epochs (default: 500)")
    
    parser.add_argument("--batch_size",
                        default=1024,
                        type=int,
                        help="batch size")
    
    parser.add_argument("--learning_rate",
                        default=0.001,
                        type=float,
                        help="learning rate")
    
    parser.add_argument("--patience",
                        default=20,
                        type=int,
                        help="early stopping")
    
    parser.add_argument("--monitor",
                        default="val_accuracy",
                        help="monitor the val_loss/loss/acc/val_acc")  
    
    parser.add_argument("--monitor_mode",
                        default="max",
                        help="min/max/auto") 
    
    parser.add_argument("--loss",
                        default='categorical_crossentropy',
                        help="loss fucntion")  
        
    
    args = parser.parse_args()
    return args

#==========generate gauss function=================#
def gaussian(x, sigma,  u):
    y = np.exp(-(x - u) ** 2 / (2 * sigma ** 2)) / (sigma * math.sqrt(2 * math.pi))
    return y/np.max(abs(y))   

# In[] main
if __name__ == '__main__':
    # In[]
    args = read_args()
    start_gpu(args)
    #====================================#
    save_path_dir='./acc_loss_fig/'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)  
    save_path_dir='./res/'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)   
    save_path_dir='./model/'
    if not os.path.exists(save_path_dir):
        os.makedirs(save_path_dir)         
        
    #==========load model=================#
    if args.model_name=='bulid_pd_model':
        print(args.model_name)
        model=build_pd_model(time_input=(400,1))
        model.summary()
    
    elif args.model_name=='cnn_ross':
        print(args.model_name)
        model=build_ross_model(time_input=(400,1))
        model.summary()
    elif args.model_name=='bulid_pp_model':
        print(args.model_name)
        model=build_PP_model(time_input=(400,1))
        model.summary()
            
    else:
        print(args.model_name)
        model=pd_model(time_input=(400,1,1))
        model.summary() 
    #==========load data=================# 
    if args.dataset=='ins':
        # if you use the instance dataset directly
        # you can uncomment the code below
        #================================================================#
        # evt_csv='./INSTANCE/dataset/metadata_Instance_events_v2.csv'
        # df = pd.read_csv(evt_csv, keep_default_na=False)
        # evt_hdn='./INSTANCE/dataset/Instance_events_counts.hdf5'
        # L=h5py.File(evt_hdn, 'r')
        
        # no_list=df[(df.trace_polarity=='undecidable')]['trace_name'].tolist() 
        # np.random.seed(7)
        # np.random.shuffle(no_list)
        # rm_list=no_list[:802858]
        # all_list=df['trace_name'].tolist() 
        # use_list=list(set(all_list) - set(rm_list)) 
        # np.random.seed(7)
        # np.random.shuffle(use_list)
        
        # batch_size=1024
        # train_list=use_list[:int(len(use_list)*0.8)]
        # gen_train=DataGenerator_PD( L,df, len(train_list),indexes=train_list,  batch_size=batch_size, tmsf=False, reve=True)
        # steps_per_epoch=len(train_list)// batch_size   
        
        # test_list=use_list[int(len(use_list)*0.8):]
        # gen_test=DataGenerator_PD( L,df, len(test_list),indexes=test_list,  batch_size=batch_size, tmsf=False, reve=False)
        # validation_steps = len(test_list)// batch_size 
        #================================================================#
        
        # if you process original dataset to scsn dataset  format
        # you can use the code below
        f5=h5py.File('./INSTANCE/ins_po_train.hdf5','r')
        f6=h5py.File('./INSTANCE/ins_po_test.hdf5','r')
        
        batch_size=1024
        train_num=len(f5['Y'])
    
        steps_per_epoch=train_num//batch_size
                
        gen_train=DataGenerator_PD_Ins( f5, train_num,  batch_size=batch_size, tmsf=False, reve=True)
        steps_per_epoch=train_num// batch_size   
        
        test_num=len(f6['Y'])
        gen_test=DataGenerator_PD_Ins( f6, test_num,  batch_size=batch_size, tmsf=False, reve=False)
        
        validation_steps = test_num// batch_size    
        
    if  args.dataset=='hinet':   
        evt_csv='./HINET_DATA/HD/HINET_Polarity_15_19.csv'
        df = pd.read_csv(evt_csv, keep_default_na=False)
        evt_hdn='./HINET_DATA/HD/HINET_Polarity_15_19.hdf5'
        L=h5py.File(evt_hdn, 'r')

        use_list=df['trace_name'].tolist() 
        np.random.seed(7)
        np.random.shuffle(use_list)
        
        batch_size=1024
        train_list=use_list[:int(len(use_list)*0.8)]
        gen_train=DataGenerator_PD_Hinet( L,df, len(train_list),indexes=train_list,  batch_size=batch_size, tmsf=False, reve=True)
        steps_per_epoch=len(train_list)// batch_size   
        
        test_list=use_list[int(len(use_list)*0.8):]
        gen_test=DataGenerator_PD_Hinet( L,df, len(test_list),indexes=test_list,  batch_size=batch_size, tmsf=False, reve=False)
        validation_steps = len(test_list)// batch_size 
        
    if  args.dataset=='scsn':     
        
        f5=h5py.File('./SCEDC_FM_DATA/scsn_p_2000_2017_6sec_0.5r_fm_train.hdf5','r')
        f6=h5py.File('./SCEDC_FM_DATA/scsn_p_2000_2017_6sec_0.5r_fm_test.hdf5','r')
        
        batch_size=1024
        train_num=len(f5['Y'])
    
        steps_per_epoch=train_num//batch_size
        if args.model_name=='bulid_pp_model':   
            gaus=gaussian(np.linspace(-5, 5, 100),1,0) 
            gen_train=DataGenerator_PP_Scsn( f5, train_num,gaus,  batch_size=batch_size, tmsf=False, reve=True)
        else:                
            gen_train=DataGenerator_PD_Scsn( f5, train_num,  batch_size=batch_size, tmsf=False, reve=True)
        steps_per_epoch=train_num// batch_size   
        
        test_num=len(f6['Y'])
        if args.model_name=='bulid_pp_model':   
            gaus=gaussian(np.linspace(-5, 5, 100),1,0)  
            gen_test=DataGenerator_PP_Scsn( f6, test_num,gaus,  batch_size=batch_size, tmsf=False, reve=False)
        else:
            gen_test=DataGenerator_PD_Scsn( f6, test_num,  batch_size=batch_size, tmsf=False, reve=False)
        
        validation_steps = test_num// batch_size 

        
    if  args.dataset=='pnw': 
        
        evt_csv='./PNW/comcat_metadata.csv'
        df = pd.read_csv(evt_csv, keep_default_na=False)
        evt_hdn='./PNW/comcat_waveforms.hdf5'
        L=h5py.File(evt_hdn, 'r')

        use_list=np.arange(np.shape(df)[0])
        np.random.seed(7)
        np.random.shuffle(use_list)
        
        batch_size=1024
        train_list=use_list[:int(len(use_list)*0.8)]
        if args.model_name=='bulid_pp_model':   
            gaus=gaussian(np.linspace(-5, 5, 100),1,0) 
            gen_train=DataGenerator_PP_PNW( L,df,gaus, len(train_list),indexes=train_list,  batch_size=batch_size, tmsf=False, reve=True)
        else:
            gen_train=DataGenerator_PD_PNW( L,df, len(train_list),indexes=train_list,  batch_size=batch_size, tmsf=False, reve=True)
            
        steps_per_epoch=len(train_list)// batch_size   
        test_list=use_list[int(len(use_list)*0.8):]
        
        if args.model_name=='bulid_pp_model':
            gaus=gaussian(np.linspace(-5, 5, 100),1,0)            
            gen_test=DataGenerator_PP_PNW( L,df,gaus, len(test_list),indexes=test_list,  batch_size=batch_size, tmsf=False, reve=False)
        else:
            gen_test=DataGenerator_PD_PNW( L,df, len(test_list),indexes=test_list,  batch_size=batch_size, tmsf=False, reve=False)
        validation_steps = len(test_list)// batch_size 

    #=====================run model============================#
    if args.model_name=='bulid_pp_model':
        args.loss=['mse','categorical_crossentropy']
        args.monitor='val_loss'
        args.monitor_mode='min'
        
    model.compile(loss=args.loss,optimizer=tf.keras.optimizers.Adam(),metrics=['accuracy'])
    saveBestModel= ModelCheckpoint('./model/%s.h5'%args.save_name, monitor=args.monitor, verbose=1, save_best_only=True,mode=args.monitor_mode)
    estop = EarlyStopping(monitor=args.monitor, patience=args.patience, verbose=0, mode=args.monitor_mode)
    callbacks_list = [saveBestModel,estop]
    
    # fit
    begin = datetime.datetime.now()
    
    history_callback=model.fit_generator(
                                generator=gen_train,    
                                steps_per_epoch=steps_per_epoch,
                                epochs=args.epochs, 
                                 verbose=1,
                                 callbacks=callbacks_list,
                                 validation_data=gen_test,
                                 validation_steps=validation_steps)    
    
    model.save_weights('./model/%s_wt.h5'%args.save_name) 
    end = datetime.datetime.now()
    print('Training time:',end-begin)
       
    #=====================rplot curve============================#
    if args.model_name=='bulid_pp_model':
        plot_loss_pp(history_callback,save_path='./acc_loss_fig/',model=args.save_name)
    else:
        plot_loss(history_callback,save_path='./acc_loss_fig/',model=args.save_name)

    #========================  evaluate ========================#
    model=load_model('./model/%s.h5'%args.save_name,custom_objects={'SeqSelfAttention':SeqSelfAttention})
    
    if args.dataset=='ins':
        gen=DataGenerator_PD_Ins( f6, test_num,  batch_size=test_num, tmsf=False, reve=False)
    if  args.dataset=='hinet':  
        gen=DataGenerator_PD_Hinet( L,df, len(test_list),indexes=test_list,  batch_size=len(test_list), tmsf=False, reve=False)
    if  args.dataset=='scsn': 
        if args.model_name=='bulid_pp_model':
            gaus=gaussian(np.linspace(-5, 5, 100),1,0)  
            gen=DataGenerator_PP_Scsn( f6, test_num,gaus,  batch_size=test_num, tmsf=False, reve=False)
        else:
            gen=DataGenerator_PD_Scsn( f6, test_num,  batch_size=test_num, tmsf=False, reve=False)
    if  args.dataset=='pnw':  
        # gen=DataGenerator_PD_PNW( L,df, len(test_list),indexes=test_list,  batch_size=len(test_list), tmsf=False, reve=False)
        if args.model_name=='bulid_pp_model':
            gaus=gaussian(np.linspace(-5, 5, 100),1,0)            
            gen=DataGenerator_PP_PNW( L,df,gaus, len(test_list),indexes=test_list,  batch_size=len(test_list), tmsf=False, reve=False)
        else:
            gen=DataGenerator_PD_PNW( L,df, len(test_list),indexes=test_list,  batch_size=len(test_list), tmsf=False, reve=False)
                
    tmp=iter(gen)
    tmp1=next(tmp)
    data=tmp1[0]['input']
    label=tmp1[1]['po']
    
    if args.model_name=='bulid_pp_model':
        pred1,pred2=model.predict(data)
    else:
        pred2=model.predict(data)
    
    file_path='./res/%s.txt'%args.save_name
    tp_up,tp_dn,tp_uw,ffp_up,ffp_dn,fp_up,fp_dn,fp_uw=confusion_matrix(pred2,label,file_path=file_path,name=args.save_name)
    np.savez('./res/%s_test_cm'%args.model_name,res=np.array([tp_up,tp_dn,tp_uw,ffp_up,ffp_dn,fp_up,fp_dn,fp_uw],dtype=object))  

      
# In[]    
    