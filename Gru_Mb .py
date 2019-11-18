#!/usr/bin/env python
# coding: utf-8

# In[56]:


import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

###################################
# Network class definition
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.gru    = nn.GRUCell(2,50) # GRU layer taking 2 inputs (L or R), has 50 units
        self.actor  = nn.Linear(50,2)  # Linear Actor layer with 2 outputs, takes GRU as input
        self.critic = nn.Linear(50,1)  # Linear Critic layer with 1 output, takes GRU as input

    def forward(self, s, h):
        h  = self.gru(s,h)  # give the input and previous hidden state to the GRU layer
        c  = self.critic(h) # estimate the value of the current state
        pi = F.softmax(self.actor(h),dim=1) # calculate the policy 
        return (h,c,pi)

net = Net()
opt = torch.optim.Adam(net.parameters(),lr=0.01) 


# In[58]:



###################################
# Run a trial

# parameters
N      = 10   # number of trials to run
T      = 20   # number of time-steps in a trial
gamma  = 0.98 # temporal discount factor

# for each trial
for n in range(N):
    
    sample  = int(np.random.uniform() < 0.5) # pick the sample input for this trial
    s_t     = torch.zeros((1,2,T))   # state at each time step
    h_t     = torch.zeros((1,50,T))  # hidden state at each time step
    h_0     = torch.zeros((1,50)) # initial hidden state
    c_t     = torch.zeros((1,T))  # critic at each time step
    R_t     = torch.zeros((1,T))  # return at each time step
    Delta_t = torch.zeros((1,T))  # difference between critic and true return at each step
    Value_l = torch.zeros((1,T))  # value loss
    
    # set the input (state) vector/tensor
    s_t[0,sample,0] = 1.0 # set first time-step stimulus
    s_t[0,0,-1]     = 1.0 # set last time-step stimulus
    s_t[0,1,-1]     = 1.0 # set last time-step stimulus
    
    # step through the trial
    for t in range(T):
            
        # run a forward step
        if t is 0:
            (h_t[:,:,t],c_t[:,t],pi) = net(s_t[:,:,t].clone(),h_0[:,:].clone())

        else:
            (h_t[:,:,t],c_t[:,t],pi) = net(s_t[:,:,t].clone(),h_t[:,:,t-1].clone())
        
    # select an action using the policy
    action = int(np.random.uniform() < pi[0,1])
    
    # compare the action to the sample
    if action is sample:
        r = 0
        print("WRONG!")
    else:
        r = 1
        print("RIGHT!")
    
   # h_t_old = h_t
   # s_t_old = s_t
    
    # step backwards through the trial to calculate gradients
    R_t[0,-1]     = r
    Delta_t[0,-1] = c_t[0,-1] - r
    Value_l[0,-1] = F.smooth_l1_loss(c_t[0,-1],R_t[0,-1]).clone()
    
    for t in np.arange(T-2,-1,-1): #backwards rollout 
        
        # calculate the return
        R_t[0,t] = gamma*R_t[0,t+1].clone()
        
        # calculate the reward prediction error
        Delta_t[0,t] = c_t[0,t] - R_t[0,t]
        
        #calculate the loss for the critic 
        crit = c_t[0,t]
        ret  = R_t[0,t]
        Value_l[0,t] = F.smooth_l1_loss(crit,ret)
        
    Vl = Value_l.sum()#calculate total loss 
    Vl.backward() #calculate the derivatives 
    opt.step() #update the weights
    opt.zero_grad() #zero gradients before next trial


# In[ ]:




