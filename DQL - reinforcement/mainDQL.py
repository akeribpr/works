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
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
import os.path
import torch
import torch.nn.parallel
import torch.optim
import os
import os.path
import torch
import torch.nn.parallel
import torch.optim
import alexnet
import numpy
from matplotlib import pyplot 
import random
import numpy.random as ran

# -*- coding: utf-8 -*-


#from __future__ import division



best_prec1 = 0

#pathV ="/more/datas/testing/testsmallReinforcement/"
pathV ="/datas/alexnetData/testsmall/"
#pathV2 ="/more/datas/testing/v2resized/"
#pathV3 ="/more/datas/testing/finalstage/"
pathV4 ="/datas/alexnetData/someoneelse/"
pathV5 ="/datas/alexnetData/resized/"



#dirs = os.listdir( path )
#dirs2 = os.listdir( path2 )
#dirs3 = os.listdir( path3 )
#dirs4 = os.listdir( path4 )


dirsV = os.listdir( pathV )
#dirsV2 = os.listdir( pathV2 )
#dirsV3 = os.listdir( pathV3 )
dirsV4 = os.listdir( pathV4 )
dirsV5 = os.listdir( pathV5 )


# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # greedy policy
GAMMA = 0.9                 # reward discount
TARGET_REPLACE_ITER = 100   # target update frequency

MEMORY_CAPACITY = 50
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n
#N_STATES = env.observation_space.shape[0]
N_STATES = 57600
#N_STATES = 8100
ENV_A_SHAPE = 0 if isinstance(env.action_space.sample(), int) else env.action_space.sample().shape     # to confirm the shape
arr = []



class Net(nn.Module):
    def __init__(self, ):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(N_STATES, 50)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.out = nn.Linear(50, N_ACTIONS)
        self.out.weight.data.normal_(0, 0.1)   # initialization
    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value


class DQN(object):

    def __init__(self):
        self.eval_net, self.target_net = Net().cuda(), Net().cuda()

        self.learn_step_counter = 0                                     # for target updating
        self.memory_counter = 0                                         # for storing memory
        self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
#        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.optimizer = torch.optim.SGD(self.eval_net.parameters(), lr=0.1)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):
        x = Variable(torch.unsqueeze(x, 0))
        # input only one sample
        if np.random.uniform() < EPSILON:   # greedy
            actions_value = self.eval_net.forward(x)
            action = torch.max(actions_value, 1)[1].data.cpu().numpy()
            action = action[0] if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)  # return the argmax index
        else:   # random
            action = np.random.randint(0, N_ACTIONS)
            action = action if ENV_A_SHAPE == 0 else action.reshape(ENV_A_SHAPE)
        return action

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
#        print('r ',r)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self,mycount):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample batch transitions
        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :N_STATES])).cuda()
        b_a = Variable(torch.LongTensor(b_memory[:, N_STATES:N_STATES+1].astype(int))).cuda()
        b_r = Variable(torch.FloatTensor(b_memory[:, N_STATES+1:N_STATES+2])).cuda()
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -N_STATES:])).cuda()

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
       
        q_next = self.target_net(b_s_).detach()     # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)   # shape (batch, 1)

        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
                
    def calculate_reward(self,s_):
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        
        return r
  
def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)   

def build_states_array():
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
    
    # load and resizing the target image
    im = Image.open('/more/datas/testing/someoneelse/frame5575.bmp')
    box = (0,20,600,480)
    im_finally = im.crop(box)
    im_finally_Resize = im_finally.resize((512,512))
    
    #convert the target image to feature vector via alexnet model    
    img_finally_tensor = preprocess(im_finally_Resize)
    img_finally_tensor.unsqueeze_(0)
    out2 = torch.autograd.Variable(img_finally_tensor).cuda()
    m_finally = model(out2).squeeze_(0)
    
    # load and resizing the input images  
    for item in dirsV:
        if os.path.isfile(pathV+item):
            im = Image.open(pathV+item)
            box = (0,20,600,480)
            a = im.crop(box)
            im_Resize = a.resize((512,512))
            im_Resize.save(pathV5+item, 'bmp')
   

    print ('finished loading and resizing images:')
    
    j = 1
    for item in dirsV5:
        if os.path.isfile(pathV5+item):
            #convert the input image to feature vector via alexnet
            #model and append it into arr
            im = Image.open(pathV5+item)
            img_tensor = preprocess(im)
            img_tensor.unsqueeze_(0)
            out = torch.autograd.Variable(img_tensor).cuda()
            m = model(out).squeeze_(0)      
            print(j)
            arr.append(m.data)
            j+=1
    
    print('finished calculating')  
    
    return m_finally

def main():
    print('start')
    y = []
    print('array size: ',len(arr))
    
    #an array of initial indexes of images to start from
    iniStates = [615, 536, 483, 421, 368, 305, 247, 179, 117, 44]
    
#    x = numpy.arange(10)
    mycount = 0
#    count_1_total = 0
    results = []
    final_results = []
    dqn = DQN()
    s_end = build_states_array()

    for i_episode in range(10):
        done = False  
        ite = 0
        
        #choosing random initial state's index
        ind = random.choice(iniStates)
        s = arr[ind]
        print('Initial index: ',ind)
        
        count_0 = 0
        count_1 = 0 
        
        print('episode:', i_episode)
        print('index  ', 'action')

        while True:
            #choosing action s
            a = dqn.choose_action(s)
            print (ite,'      ->') if a == 1 else print (ite,'      <-')
    
            #if the action is right, going to the next frame.
            #else doing the oposite
            if a == 1:
                ind += 1
            else:
                ind -= 1
            #the next state is now the next frame or the previous one
            s_ = arr[ind]
            
            r2 = cos_sim(torch.autograd.Variable(s_end),torch.autograd.Variable(s))

            # getting reward 100 if done. else getting 0
            r = 0
            if r2 > 0.885:
                done = True
                r = 100
            
            #getting random action a and reward r if we are in the pre learning mode
            if dqn.memory_counter < MEMORY_CAPACITY:  
                a = random.randrange(0,2)
                r = ran.uniform(0,1)
                
            #count left or right moves
            if a == 0:
                count_0 += 1
            else:
                count_1 +=1  
            #store the transition for learning
            dqn.store_transition(s, a, r, s_)
            if dqn.memory_counter == MEMORY_CAPACITY:
                dqn.learn(mycount)
            
            if dqn.memory_counter > MEMORY_CAPACITY:
                y.append(r2)
                dqn.learn(mycount)
                if done:
                    print('done ',ite,' moves')
                    print('lefts count ', count_0)
                    print('right count ', count_1)
                    results.append(count_0+count_1)
                    final_results.append([count_0,count_1])
#                    final_results.append([len(y),count_0,count_1])           

            if done:
                mycount+=1
                break
            s = s_
            ite+=1
    print('results: left and right count array')
    print(final_results) 
    x = numpy.arange(len(y))       
    fig = pyplot.figure()
    ax = fig.add_subplot(111)
    ax.set_ylim(0.86,0.92)
    pyplot.plot(x,y)
#    pyplot.savefig('results/test2.png')
#    pyplot.savefig('results/test1.png')
    pyplot.show()
#    print(results)
#    print('total_0: ',count_1_total)
            
if __name__ == '__main__':
    main()