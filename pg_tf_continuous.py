# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 13:31:58 2019

@author: Aidin
"""

import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from FeatureTransformer import FeatureTransformer

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
        self.ft = ft
        self.layers = []
        M1 = D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1,M2)
            self.layers.append(layer)
            M1 = M2
            
            
        
        self.mean_layer = HiddenLayer(M1, 1, lambda x: x,use_bias= False, zeros = True)
        self.stdv_layer = HiddenLayer(M1, 1, tf.nn.softplus,use_bias= False, zeros = False)
        
        
        
        self.X = tf.placeholder(tf.float32,shape = (None , D), name = 'X')
        self.actions = tf.placeholder(tf.float32,shape = (None ,), name = 'actions')
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
            self.costs.append(cost)
            
            
    def predict(self,X):
            X = np.atleast_2d(X)
            X = self.ft.transform(X)
            return self.session.run(self.predict_op, feed_dict={self.X : X})
        
def play_one_episode(env,pmodel,vmodel,gamma):
    observation = env.reset()
    done = False
    totalreward = 0
    iters = 0
        
    while not done and iters <2000:
        action = pmodel.sample_action(observation)
        prev_observation = observation
        observation, reward, done, info = env.step([action])
        
        
        totalreward +=reward
        V_next = vmodel.predict(observation)
        G = reward + gamma*V_next
        advantage = G -vmodel.predict(prev_observation)
        pmodel.partial_fit(prev_observation,action,advantage)
        vmodel.partial_fit(prev_observation, G)
        iters +=1
    
    
    return totalreward, iters
    
    
def plotAvgReward(totalRewards):
    N = len(totalRewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalRewards[max(0, t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()
    
    
    
def main():
    env = gym.make('MountainCarContinuous-v0')
    ft = FeatureTransformer(env,n_components = 100)
    D = ft.dimensions
    
    pmodel = PolicyModel(D,ft,[])
    vmodel = ValueModel(D,ft,[])
    init = tf.global_variables_initializer()
    
    session = tf.InteractiveSession()
    session.run(init)
    pmodel.set_session(session)
    vmodel.set_session(session)
    gamma = 0.95
    
    N = 500
    
    
    totalrewards = np.empty(N)
    
    for n in range(N):
        totalrewards[n], num_steps = play_one_episode(env, pmodel,vmodel, gamma)
#        if (n + 1) % 100 == 0:
        print("episode:", n+1, "total reward: %.1f" % totalrewards[n],"num steps: %d" %num_steps)
    
    print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
    print("total steps:", -totalrewards.sum())
        
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plotAvgReward(totalrewards)
    
    
if __name__ == '__main__':
    main()