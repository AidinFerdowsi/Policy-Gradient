# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 18:17:44 2019

@author: Aidin
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt



class HiddenLayer:
    def __init__(self,M1,M2, f = tf.nn.tanh , use_bias = True ):
        self.W = tf.Variable(tf.random_normal(shape= (M1,M2)))
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        self.f = f
        
    def forward(self,X):
        if self.use_bias:
            a = tf.matmul(X,self,W) + self.b
            
        else:
            a = tf.matmul(X,self,W)
        
        return self.f(a)
    
    
    
class PolicyModel:
    
    def __init__(self, D, K, hidden_layer_sizes):
        
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1 = M2
            
            
        
        
        layer = HiddenLayer(M1,K, tf.nn.softmax,use_bias= False)
        self.layers.append(layer)
        
        
        self.X = tf.placeholder(tf.float32,shape = (None , D), name = 'X')
        self.actions = tf.placeholder(tf.float32,shape = (None , D), name = 'actions')
        self.advantages = tf.placeholder(tf.float32,shape = (None , D), name = 'advantages')
        
        
        Z = self.X
        
        for layer in self.layers:
            Z = layer.forward(Z)
            
        self.predict_op = Z
                
        
        