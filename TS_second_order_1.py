# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 16:18:39 2015

@author: dsing001
"""

from NN_2 import MyFirstNN
import numpy
import math
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 100,fnc='sigmoid'): 
    

    n = MyFirstNN(n_hidden_layer,fnc=fnc,learning_eta=learning_rate,epochs=epochs,Normalize=True,batch_size = 88,outer_fnc='linear',wgt_decay=0.001) 

    train_err,test_err,wghts_after_each_epoch = n.fit(train_data,train_cls,test_data,test_cls)
    print train_err
    print
    print test_err
    print
    train_error= pd.Series(numpy.array(train_err).flat)
    test_error = pd.Series(numpy.array(test_err).flat)
    

    #print wghts_after_each_epoch
    print ('predict')
    predicted = n.predict(test_data)
    td = numpy.hstack((train_data[:,0],predicted[:,0]))
    print td.shape
    td = pd.Series(td[400:])
    print td
    plt.figure()
    td.plot()
    td_orig = numpy.hstack((train_data[:,0],test_data[:,0]))
    td_orig = pd.Series(td_orig[400:])
    td_orig.plot()
    print td_orig
    
    plt.show()
    plt.figure()
    train_error.plot()
    test_error.plot()
    plt.show()
    
    
    

    #print wghts_after_each_epoch
    
ts_f = open('./ts.txt')
ts = []
for line in ts_f:
    ts.append(float(line.strip('\n').strip()))


ts_f.close()

ts_t = []
k = 3
for i in range(len(ts)):
    if i <= k:
        continue
    else:

        ab = ts[i-k:i+1]

        ts_t.append(ab)
    
ts_t = np.array(ts_t)

train_data = ts_t[:400,:k]
train_class = numpy.array(ts_t[:400,k])

test_data = ts_t[400:,:k]
test_class = numpy.array(ts_t[400:,k])

td = pd.Series(ts[411:])
#td.plot()
print

print (datetime.datetime.now())
train_neural_net(train_data,train_class,test_data,test_class,n_hidden_layer=2,learning_rate=0.75,epochs =1)
print (datetime.datetime.now())