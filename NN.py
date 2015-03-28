# -*- coding: utf-8 -*-
"""
Created on Sun Nov 02 19:04:21 2014

@author: Darshan Singh

This is single hidden layer model only. It can be made generic to have as many as hidden layer and finally sum of signals at each node can be calculated 
using numpy matrix multiplication to make things bit easy.
"""
import numpy as ma
import numpy
#from numpy import ma 
import math
import datetime


def sigmoid(vl):
    return (1/(1+ma.exp(-vl)))

def sigmoid_diff(vl):
    return sigmoid(vl)* (1-sigmoid(vl))

def linear(x):
    return x
    
def linear_diff(x):
    return 1
    
class MyFirstNN:
    def __init__(self,n_hidden_layer,fnc='sigmoid',learning_eta= 1.0,epochs= 1000,outer_fnc= None,Normalize= True,batch_size = 1,wgt_decay = 0,bs=0,out_bs=0):
        self.n_hidden_layer = n_hidden_layer
        self.bs = bs
        self.out_bs = out_bs
        print n_hidden_layer
        self.trgts_dict= {}
        self.trgts_dict_pos= {}   
        self.n_in_layer = 2
        self.Normalize=Normalize
        self.n_outer_layer = 1
        self.mn= 0
        self.mx = 1
        self.min=None
        self.max = None
        self.wgt_decay = wgt_decay
        ''' You might want to use a different function at different hidden or output node '''
   
        self.fnc_hidden_layers = sigmoid
        self.fnc_diff_hidden_layers = sigmoid_diff
        self.fnc_output_layers = sigmoid
        self.fnc_diff_output_layers = sigmoid_diff

        if outer_fnc ==None:
            outer_fnc= fnc
            
        if outer_fnc=='sigmoid':
            self.fnc_output_layers = sigmoid
            self.fnc_diff_output_layers = sigmoid_diff
        else:
            self.fnc_output_layers = linear
            self.fnc_diff_output_layers = linear_diff            

            
        self.fnc_output_layers= numpy.vectorize(self.fnc_output_layers)
        self.fnc_diff_output_layers= numpy.vectorize(self.fnc_diff_output_layers)
        self.fnc_hidden_layers= numpy.vectorize(self.fnc_hidden_layers)        
        self.fnc_diff_hidden_layers= numpy.vectorize(self.fnc_diff_hidden_layers)        
        self.learning_eta = learning_eta            
        self.epochs = epochs
        self.training_error= []
        self.batch_size = batch_size

        
    def init_weights(self):
        ###create separate weight matrix for input and hidden
        #np.random.normal(mu, sigma, 1000)
        self.w_in_to_hidden = numpy.random.normal(0,0.001,self.n_hidden_layer*self.n_in_layer).reshape(self.n_hidden_layer,self.n_in_layer)
        self.delta_w_in_to_hidden = ma.zeros((self.n_hidden_layer,self.n_in_layer))
        self.w_hidden_to_out = numpy.random.normal(0,0.001,(self.n_hidden_layer + 1)*self.n_outer_layer).reshape(self.n_outer_layer,(self.n_hidden_layer + 1))
        self.delta_w_hidden_to_out = ma.zeros((self.n_outer_layer,(self.n_hidden_layer + 1)))
   
    def nrmlz(self,X):
       
        return numpy.multiply(numpy.divide(numpy.subtract(X,self.min),(self.max-self.min)),(self.mx-self.mn))

    def forward_pass(self,inpt):

        x = self.w_in_to_hidden.dot(inpt.T)
        hidden_layer_out = self.fnc_hidden_layers(x)
        #print (hidden_layer_out.shape)
   
        inputsize = hidden_layer_out.shape[1]
        bias = ma.multiply(ma.ones(inputsize).reshape(1,inputsize),self.out_bs)
        hidden_layer_out= ma.array(ma.vstack((hidden_layer_out,bias)))

        hidden_layer_diff = self.fnc_diff_hidden_layers(x)
        hidden_layer_diff= ma.vstack((hidden_layer_diff,bias))
        ''' visit all output nodes and calculate the outputs which will then be compared with target to get error at that output node '''
        x = self.w_hidden_to_out.dot(hidden_layer_out)
   
    
        out_layer_output = ma.array(self.fnc_output_layers(x))
 
        out_layer_diff = ma.array(self.fnc_diff_output_layers(x))
        return (hidden_layer_out,hidden_layer_diff,out_layer_output,out_layer_diff)
        
        
        
    def backpropagate(self,hidden_layer_out,hidden_layer_diff,out_layer_output,out_layer_diff,out,inpts):
        
        ''' betas for output layer neuron to calculate the weights '''   
        out_layer_betas = ma.multiply(out_layer_diff,(ma.subtract(out.T,out_layer_output).T).T)   
        self.delta_w_hidden_to_out = ma.multiply(self.learning_eta,out_layer_betas.dot(hidden_layer_out.T))
        ''' calculate hidden layer betas and  update input to hidden layer weights based on these betas '''
        hidden_layer_betas = ma.multiply(hidden_layer_diff,self.w_hidden_to_out.T.dot(out_layer_betas))        
        self.delta_w_in_to_hidden = ma.multiply(self.learning_eta,hidden_layer_betas.dot(inpts))


       
    def update_weights(self):
        ''' update in to hidden weights '''
 
        self.w_in_to_hidden = (1- self.wgt_decay*self.learning_eta) * self.w_in_to_hidden + self.delta_w_in_to_hidden[:-1,:]

        ''' update hidden to out weights '''
        self.w_hidden_to_out = self.w_hidden_to_out + self.delta_w_hidden_to_out

    #call this method to determine the input layers as well as the output layers and distinct targets
    def binarizelabels(self):
        
        dist_targets = self.labels       
        trgts_dict = {}
        trgts_dict_pos = {}
        for i,target in enumerate(sorted(dist_targets)):
            trgts_dict[target] = i
            trgts_dict_pos[i] = target
        self.trgts_dict = trgts_dict
        self.trgts_dict_pos = trgts_dict_pos
        self.n_outer_layer = len(dist_targets)
       
   
    def fit(self,X,Y,test_data=None,test_class=None):          
        trgts = Y  
     
        if self.Normalize:
            if self.min==None:
                self.min = X.min(axis=0)
            if self.max==None:
                self.max = X.max(axis=0)
            X = self.nrmlz(X)
            
        self.n_in_layer = X.shape[1] + 1 # add a bias

        self.init_weights()
        bs = self.batch_size
        wghts_after_each_epoch = [] 
        inputsize = X.shape[0]
        bias = ma.multiply(ma.ones(inputsize).reshape(inputsize,1),self.bs)
        X= ma.hstack((X,bias))
        tst_error = []
        for epoch in range(self.epochs):  
            print epoch
            
            wghts_after_each_epoch.append((self.w_hidden_to_out,self.w_in_to_hidden))
          
            error= ma.zeros(self.n_outer_layer,dtype='float64')  
            
            #self.learning_eta *= 0.9
            for i in range(int(math.ceil(X.shape[0]*1.0/bs))):     
                inputs = X[i*bs:(i+1)*bs,:]   
                targets = trgts[i*bs:(i+1)*bs]     
                out = targets 
                hidden_layer_out,hidden_layer_diff,out_layer_output,out_layer_diff = self.forward_pass(inputs)   
                self.backpropagate(hidden_layer_out,hidden_layer_diff,out_layer_output,out_layer_diff,out,inputs)      
                self.update_weights()                             
                error = error + ma.sum(ma.subtract(targets,out_layer_output.T)**2,axis=0)  
                
            train_error = error/(X.shape[0])
  
            print 'train_error',train_error
            self.training_error.append(train_error)
            print (epoch, str(datetime.datetime.now()) ,'end')
            if test_data != None:
                 predicted = self.predict(test_data)
                
                 test_cls= numpy.array(test_class).reshape(test_data.shape[0],self.n_outer_layer)
                 
                 test_error = ma.sum(ma.power(test_cls-predicted,2),axis=0)/(test_data.shape[0])
                 print 'test_error',test_error
                 tst_error.append(test_error)
          
                       
                
                
        train_err = self.training_error
        return(train_err,tst_error,wghts_after_each_epoch)

    def predict(self,X):

       if self.Normalize:
            X = self.nrmlz(X)
   
       inputsize = X.shape[0]
       bias = ma.multiply(ma.ones(inputsize).reshape(inputsize,1),self.bs)
       X   = ma.hstack((X,bias))

      
       hidden_layer_out,hidden_layer_diff,out_layer_output,out_layer_diff = self.forward_pass(X) 
     
       tst_cls_out = out_layer_output.T

       return tst_cls_out
       

        
                
if __name__ == "__main__":
    def train_neural_net(train_data,train_cls,test_data,test_cls,learning_rate=0.7,epochs = 100,fnc='sigmoid'):
        n_hidden_layer = 2

        n = MyFirstNN(n_hidden_layer,fnc=fnc,learning_eta=learning_rate,epochs=epochs,outer_fnc='linear',batch_size= 4)  
        train_err,wghts_after_each_epoch = n.fit(train_dt,train_cls)      
        print train_err
        print wghts_after_each_epoch
        print
        print 'test'
        predicted = n.predict(test_dt)
        print predicted
  
    
    train_dt = numpy.array([[1,0],[1,1],[0,1],[0,0]])
    train_cls = [0,1,0,1]
    test_dt = train_dt
    test_cls = train_cls
    train_neural_net(train_dt,train_cls,test_dt,test_cls,epochs =2)
        
    
        
        
            
        
            