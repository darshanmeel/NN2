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
from matplotlib.backends.backend_pdf import PdfPages

def train_neural_net(train_data,train_cls,test_data,test_cls,n_hidden_layer= 5,learning_rate=1,epochs = 100,fnc='sigmoid',wgt_decay=0.0,pp=None): 
    

    n = MyFirstNN(n_hidden_layer,fnc=fnc,learning_eta=learning_rate,epochs=epochs,Normalize=True,batch_size = 1,outer_fnc='linear',wgt_decay=wgt_decay,bs = 0.0,out_bs=0.0) 

    train_err,test_err,wghts_after_each_epoch = n.fit(train_data,train_cls,test_data,test_cls)

    print 'train error'
    print n_hidden_layer,learning_rate,wgt_decay
    train_error= pd.Series(numpy.array(train_err).flat)
    print train_error[epochs -1]
    test_error = pd.Series(numpy.array(test_err).flat)
    print test_error[epochs -1]
    #show the train and test erros
    print 'show the train and test errors starts'
    plt.show()
    fig = plt.figure(figsize=(12, 12))
    train_error.plot(label = 'Training Error')
    test_error.plot(label = 'Test Error')
    plt.legend() 
    ttl = 'Train vs Test Error for learning rate ' + str(learning_rate) + ' and with ' + str(n_hidden_layer) + ' hidden layers and smoothing rate ' + str(wgt_decay)
    fig.suptitle(ttl, fontsize=12)
    
    #plt.show()
    plt.savefig(pp, format='pdf')
    print 'show the train and test errors ends'
    #just try on train data and see what happens
    print 'train_data fit starts'
    predicted = n.predict(train_data)
    #print predicted,test_cls
    td = numpy.array(predicted[:,0])

    td = pd.Series(td)
    #print td
    fig=plt.figure(figsize=(12, 12))
    td.plot(label='Predicted Training Data')
    td_orig = numpy.array(train_cls)
    td_orig = pd.Series(td_orig)
    td_orig.plot(label='Original Training Data')
    plt.legend() 
    ttl = 'Original Training vs Predicted Training data for learning rate ' + str(learning_rate) + ' and with ' + str(n_hidden_layer) + ' hidden layers and smoothing rate ' + str(wgt_decay)
    fig.suptitle(ttl, fontsize=12)
    #plt.show()
    plt.savefig(pp, format='pdf')
    print 'train_data fit ends'
    #print wghts_after_each_epoch
    print 'test_data fit starts'
    predicted = n.predict(test_data)
    #print predicted,test_cls
    td = predicted[:,0]
    td = pd.Series(td)
    #print td
    fig = plt.figure(figsize=(12, 12))
    td.plot(label='Predicted Test Data')
    td_orig = numpy.array(test_cls)
    td_orig = pd.Series(td_orig)
    td_orig.plot(label='Original Test Data')
    plt.legend() 
    ttl = 'Original Test vs Predicted Test data  for learning rate ' + str(learning_rate) + ' and with ' + str(n_hidden_layer) + ' hidden layers and smoothing rate ' + str(wgt_decay)
    fig.suptitle(ttl, fontsize=12)
    #plt.show()
    plt.savefig(pp, format='pdf')
    #print td_orig
    print 'test_data fit ends'
  
    
    
    

    #print wghts_after_each_epoch
    
ts_f = open('./ts.txt')
ts = []
for line in ts_f:
    ts.append(float(line.strip('\n').strip()))


ts_f.close()
ln = len(ts)
#add some random noise to the TS
rdm = numpy.random.normal(0,0.05,ln)
#ts = numpy.array(ts)+ rdm


ts_t = []
k = 10
numoftrn = 450
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

smthing=True
# I havekept the number of epochs to be 100 and I have not used any early stopping.
print (datetime.datetime.now())
pp = PdfPages('Output_figures.pdf')
# you can change the number of hidden layers for example range(6,9,2) means that try with 6 hidden layers and then increase it by 2 untill you cross 9 this means it will run for hidden layers 6 and 8
for nl in range(6,9,2):
    n_hidden_layer= nl
    # eta ia learning rate and if you see that it is range between 0 and 100 and this is because range function takes integers only. I have divided this by 100 to make it proper learning rates as learning
    # rate should be between 0 and 1.
    for eta in range(75,76,15):
        learning_rate= eta/100.0
        if (smthing):
            smt = range(0,5,3)
        else:
            smt = range(1)
        # smoothing rate 0 means no regularization otherwise it means that apply regularization. Here as well values are between 0 and 5 but these are divided by 10000.
        for smts in smt:      
            train_neural_net(train_data,train_class,test_data,test_class,n_hidden_layer=n_hidden_layer,learning_rate=learning_rate,epochs =100,wgt_decay=smts/10000.0,pp=pp)
pp.close()      
print (datetime.datetime.now())