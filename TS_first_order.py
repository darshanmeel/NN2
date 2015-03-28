# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:18:39 2015

@author: dsing001
"""

from NN import MyFirstNN
import numpy
import math
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 100,fnc='sigmoid'): 
    

    n = MyFirstNN(n_hidden_layer,fnc=fnc,learning_eta=learning_rate,epochs=epochs,Normalize=True,batch_size = 1,outer_fnc='linear',wgt_decay=0.0000,bs = 0.0,out_bs=0.0) 

    train_err,test_err,wghts_after_each_epoch = n.fit(train_data,train_cls,test_data,test_cls)
    print train_err
    print
    print test_err
    print
    train_error= pd.Series(numpy.array(train_err).flat)
    test_error = pd.Series(numpy.array(test_err).flat)
    #show the train and test erros
    print 'show the train and test errors starts'
    plt.show()
    plt.figure()
    train_error.plot()
    test_error.plot()
    plt.show()
    print 'show the train and test errors ends'
    #just try on train data and see what happens
    print 'train_data fit starts'
    predicted = n.predict(train_data)
    #print predicted,test_cls
    td = numpy.array(predicted[:,0])
    print td.shape
    td = pd.Series(td)
    #print td
    plt.figure(figsize=(10, 10))
    td.plot()
    td_orig = numpy.array(train_cls)
    td_orig = pd.Series(td_orig)
    td_orig.plot()
    #print td_orig
    
    plt.show()
    print 'train_data fit ends'
    #print wghts_after_each_epoch
    print 'test_data fit starts'
    predicted = n.predict(test_data)
    #print predicted,test_cls
    td = numpy.hstack((train_data[:,0],predicted[:,0]))
    print td.shape
    td = pd.Series(td[400:])
    #print td
    plt.figure()
    td.plot()
    td_orig = numpy.hstack((train_data[:,0],test_cls))
    td_orig = pd.Series(td_orig[400:])
    td_orig.plot()
    #print td_orig
    print 'test_data fit ends'
  
    
    
    

    #print wghts_after_each_epoch
    
ts_f = open('./ts.txt')
ts = []
for line in ts_f:
    ts.append(float(line.strip('\n').strip()))


ts_f.close()

ts_t = []
k = 10
numoftrn = 300
for i in range(len(ts)):
    if i <= k:
        continue
    else:

        ab = ts[i-k:i+1]

        ts_t.append(ab)
    
ts_t = np.array(ts_t)

train_data = ts_t[:numoftrn,:k]
train_class = numpy.array(ts_t[:numoftrn,k])

test_data = ts_t[numoftrn:,:k]
test_class = numpy.array(ts_t[numoftrn:,k])



print (datetime.datetime.now())
train_neural_net(train_data,train_class,test_data,test_class,n_hidden_layer=8,learning_rate=0.9,epochs =100)
print (datetime.datetime.now())