# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:31:58 2019

@author: Aidin
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class HiddenLayer:
    def __init__(self,M1,M2, f = tf.nn.tanh , use_bias = True, zeros = False ):
        if zeros:
            W = np.zeros((M1,M2), dtype = np.float32)
        else:
            W = tf.random_normal(shape= (M1,M2))     
        self.W = tf.Variable(W)
        
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
        self.f = f
        
    def forward(self,X):
        if self.use_bias:
            a = tf.matmul(X,self.W) + self.b
            
        else:
            a = tf.matmul(X,self.W)
        
        return self.f(a)
    
    
class PolicyModel:
    
    def __init__(self, D, ft, hidden_layer_sizes = []):
        
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1 = M2
            
            
        
        self.mean_layer = HiddenLayer(M1, 1, lambda x: x,use_bias= False, zeros = True)
        self.stdv_layer = HiddenLayer(M1, 1, tf.nn.softplus,use_bias= False, zeros = False)
        
        
        
        self.X = tf.placeholder(tf.float32,shape = (None , D), name = 'X')
        self.actions = tf.placeholder(tf.int32,shape = (None ,), name = 'actions')
        self.advantages = tf.placeholder(tf.float32,shape = (None ,), name = 'advantages')
        
        
        Z = self.X
        
        for layer in self.layers:
            Z = layer.forward(Z)
            
        mean = self.mean_layer.forward(Z)
        stdv = self.stdv_layer.forward(Z) + 1e-5
        
        mean = tf.reshape(mean,[-1])
        stdv = tf.reshape(stdv,[-1])
        
        norm = tf.contrib.distributions.Normal(mean,stdv)
        
        self.predict_op = tf.clip_by_value(norm.sample(),-1,1)
                    
            
        log_probs = norm.log_prob(self.actions)
        
            
        cost = -tf.reduce_sum(self.advantages*log_probs + 0.1*norm.entropy())
            
            
        self.train_op = tf.train.AdamOptimizer(10e-3).minimize(cost)
            
            
    def set_session(self,session):
            self.session = session
            
        
        
    def partial_fit(self,X,actions,advantages):
            X = np.atleast_2d(X)
            X = self.ft.transform(X)
            
            
            actions = np.atleast_1d(actions)
            advantages = np.atleast_1d(advantages)
            self.session.run(
                    self.train_op,
                    feed_dict = {
                            self.X : X,
                            self.actions : actions,
                            self.advantages : advantages,
                            }
                    )
            
    def predict(self,X):
            X = np.atleast_2d(X)
            X = self.ft.transform(X)
            return self.session.run(self.predict_op, feed_dict={self.X : X})
        
        
    def sample_action(self,X):
            p = self.predict(X)[0]
            return p
        
        
class ValueModel:
    def __init__(self, D, ft, hidden_layer_sizes = []):
        self.ft = ft
        self.costs = []
        self.layers = []
        
        
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1 = M2
            
            
        
        
        layer = HiddenLayer(M1,1, lambda x:x)
        self.layers.append(layer)
        
        
        self.X = tf.placeholder(tf.float32,shape = (None , D), name = 'X')
        self.Y = tf.placeholder(tf.float32,shape = (None , ), name = 'Y')
        
        Z = self.X
        
        for layer in self.layers:
            Z = layer.forward(Z)
            
        Y_hat = tf.reshape(Z,[-1])
        self.predict_op = Y_hat
        
        cost = tf.reduce_sum(tf.square(self.Y - Y_hat))
        self.cost = cost
        self.train_op = tf.train.AdamOptimizer(1e-1).minimize(cost)
        
    def set_session(self,session):
            self.session = session
            
        
        
    def partial_fit(self,X, Y):
            X = np.atleast_2d(X)
            X = self.ft.transform(X)
            Y = np.atleast_1d(Y)
            self.session.run(self.train_op,feed_dict = {self.X : X, self.Y : Y})
            cost = self.session.run(self.cost,feed_dict = {self.X : X, self.Y : Y})
            
            
    def predict(self,X):
            X = np.atleast_2d(X)
            return self.session.run(self.predict_op, feed_dict={self.X : X})