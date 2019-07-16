"""
View more, visit my tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
torch: 0.3
gym: 0.8.1
numpy
"""
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import gym
import argparse
import torch.nn.init
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from PIL import Image
import os.path
import torch
import torch.nn.parallel
from  torch.nn.modules import distance
import torch.optim
import os
import torch.utils.data as data
import os.path
import torch
import torch.nn.parallel
import torch.optim
import alexnet


# -*- coding: utf-8 -*-



#from __future__ import division



best_prec1 = 0

pathV1 ="/datas/ActionsData/left/"
pathV2 ="/datas/ActionsData/right/"
pathV3 ="/datas/ActionsData/leftResized/"
pathV4 ="/datas/ActionsData/rightResized/"

dirsV1 = os.listdir( pathV1 )
dirsV2 = os.listdir( pathV2 )
dirsV3 = os.listdir( pathV3 )
dirsV4 = os.listdir( pathV4 )



# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency

MEMORY_CAPACITY = 150
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
#N_STATES = env.observation_space.shape[0]
N_STATES = 57600
#N_STATES = 8100
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
arr = []
arr1 = []
arr2 = []



def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Linear(N_STATES * 2, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, N_ACTIONS)
            )
    def forward(self, x):
        nn.Dropout(),
        x = self.features(x)
        return x

def build_arrays():
       
    
    print('start')
    global args, best_prec1
    
    # load a pretrained alexnet model
    
    model = alexnet.alexnet(True)

    model.cuda()
    
    model.classifier = nn.Sequential(
    )
    
    #preprocessing
    preprocess = transforms.Compose([
    transforms.Pad(1),
    transforms.ToTensor()
    
    ])
    
#    #resizing images
    
#    for item in dirsV1:
#        if os.path.isfile(pathV1+item):
#            im = Image.open(pathV1+item)
#            box = (0,20,600,480)
#            a = im.crop(box)
#            im_Resize = a.resize((512,512))
#            im_Resize.save(pathV3+item, 'bmp')
# 
#    print ('finished loading and resizing images left :')
#    
#    for item in dirsV2:
#        if os.path.isfile(pathV2+item):
#            im = Image.open(pathV2+item)
#            box = (0,20,600,480)
#            a = im.crop(box)
#            im_Resize = a.resize((512,512))
#            im_Resize.save(pathV4+item, 'bmp')
#            
#    print ('finished loading and resizing images right :')
    
#   appending the feature's vectors of the left images into arr1, and the right into arr2
    j = 1
    for item in dirsV3:
        if os.path.isfile(pathV3+item):
            im = Image.open(pathV3+item)
            img_tensor = preprocess(im)
            img_tensor.unsqueeze_(0)
            out = torch.autograd.Variable(img_tensor).cuda()
            m = model(out).squeeze_(0)
            print('appending left: ',j)
            arr1.append(m.data)
            j+=1
    print('len of left: ',len(arr1))
    
    j = 1
       
    for item2 in dirsV4:
        if os.path.isfile(pathV4+item2):
            im = Image.open(pathV4+item2)
            img_tensor = preprocess(im)
            img_tensor.unsqueeze_(0)
            out = torch.autograd.Variable(img_tensor).cuda()
            m = model(out).squeeze_(0)
            print('appending right: ',j)

            arr2.append(m.data)
            j+=1
            
    print('len of right: ',len(arr2))

    print('finished appending.') 


def main():
    print('start')   
    build_arrays()

    # define and init the net
    new_net = Net().cuda()
    new_net.apply(init_weights)
    
    #define the optimizer of the net
    optimizer_policy_new = torch.optim.Adam(new_net.parameters(), lr= 0.01)
        
    #define the loss function
    loss_func = nn.MSELoss()
    
    #define two arrays with one target tensor
    array1 = torch.cuda.FloatTensor([[0.0 , 100.0]])
    array2 = torch.cuda.FloatTensor([[100.0 , 0.0]])

    #define new empty output array
    outputFinal =  torch.cuda.FloatTensor([])
    #define new empty target array
    targetFinal =  torch.cuda.FloatTensor([])

    #learning the network to tag left and rights inputs trough 2221 features vectors
    for j in range (2221):   
        # insert into in1 two features vectors that represent a left move
        in1 = torch.cat((arr1[int(j)], arr1[int(j) + 1]), 0)
        print('len', len(in1))
        #the network tags the move
        output = new_net(in1)
        output = output.unsqueeze_(0)
        print('output', output)
        #adds the output and the target into the arrays
        outputFinal = torch.cat((outputFinal,output), 0)
        targetFinal = torch.cat((targetFinal,array1), 0)
        
        # insert into in2 two features vectors that represent a right move
        in2 = torch.cat((arr2[int(j)], arr2[int(j) + 1]), 0)
        #the network tags the move
        output = new_net(in2)
        output = output.unsqueeze_(0)
        #adds the output and the target into the arrays
        outputFinal = torch.cat((outputFinal,output), 0)
        targetFinal = torch.cat((targetFinal,array2), 0)

        #for batch size 16 do backpropogetion   
        if j % 15 == 0 and j > 1 and j <2221:
            loss = loss_func(outputFinal, targetFinal)
            print(outputFinal, targetFinal)
            optimizer_policy_new.zero_grad()
            loss.backward()
            optimizer_policy_new.step()
            outputFinal =  torch.cuda.FloatTensor([])
            targetFinal =  torch.cuda.FloatTensor([])
            print('loss ' , loss)
            
            
        
            
if __name__ == '__main__':
    main()# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

