#!/usr/bin/env python
"""Run Atari Environment with DQN."""
import argparse
import os
import random
import gym
import time
from builtins import input
import numpy as np
import tensorflow as tf
from keras.layers import (Activation, Convolution2D,  Dense, Flatten, Input,
                          Permute, Lambda)
from keras.layers.convolutional import Conv2D
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
#from keras.utils import plot_model
from keras import backend as K
from gym import wrappers
from PIL import Image
import matplotlib.image as mpimg
from collections import deque
from scipy.misc import imresize, imsave

def create_model(window, input_shape, num_actions,
                 model_name='q_network'):  # noqa: D103
    
    #print('The input shape is', input_shape)
    input_shapes=(input_shape[0],input_shape[1],window)
    inputs = Input(shape=input_shapes)
    model = Sequential()
    #Should add strides here, it is missing. Added strides
    x=Convolution2D(filters=32,kernel_size=(8, 8),strides=4,activation='relu')(inputs)
    x=Convolution2D(filters=64,kernel_size=(4, 4),strides=2,activation='relu')(x)
    x=Convolution2D(filters=64,kernel_size=(3, 3),strides=1,activation='relu')(x)
    x=Flatten()(x)
    fc1=Dense(512, activation='relu')(x)
    fc2=Dense(512, activation='relu')(x)
    mid1=Dense(num_actions)(fc1)
    mid2=Dense(1)(fc2)
    mid1=Lambda(lambda x: x - K.mean(x, keepdims=True))(mid1)
    outs = Lambda(lambda x: x[0]+x[1])([mid1,mid2])
    model = Model(inputs=inputs, outputs=outs)
    #print(model.summary())
    
    return model


def get_output_folder(parent_dir, env_name):
    
    os.makedirs(parent_dir, exist_ok=True)
    experiment_id = 0
    for folder_name in os.listdir(parent_dir):
        if not os.path.isdir(os.path.join(parent_dir, folder_name)):
            continue
        try:
            folder_name = int(folder_name.split('-run')[-1])
            if folder_name > experiment_id:
                experiment_id = folder_name
        except:
            pass
    experiment_id += 1

    parent_dir = os.path.join(parent_dir, env_name)
    parent_dir = parent_dir + '-run{}'.format(experiment_id)
    return parent_dir

def get_frames(env, action, imsize):
    observation, reward, done, info =env.step(action)
    newobs = preprocess(observation, imsize)
    newobs = np.stack((newobs,newobs), axis=-4)  
    return newobs

def rgb2gray(image):
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def preprocess(img, imsize):
    result = rgb2gray(img)
    result = imresize(result,(imsize[0],imsize[1]))
    imsave('initial.png', result)
    #print('The size of the result is',result.size) 
    newres = np.stack((result,result,result,result),axis=2)
    return newres

##We will define all the policies that we use here
class policy():
    epsilon = 1
    decr = 0.000009
    def __init__(self):
        epsilon = 1
        decr = 0.000009 #as described in the paper #doubt1 too high?
    def uniformpolicy(num):
        return np.random.randint(0, num)
    def greedyepsilon(self,epsilon, qvalues):
        val = random.random()
        #print('The random value now is', val)
        if val <= self.epsilon:
            act = random.randint(0,len(qvalues[0])-1)
            #print('in the random function', act, qvalues, len(qvalues))
            return act
        else:
            return np.argmax(qvalues)

    def ldgreedyepsilon(self,qvalues):
        if self.epsilon>0.1:
            self.epsilon = self.epsilon - self.decr
        else:
            self.epsilon=0.1
        return self.greedyepsilon(self.epsilon,qvalues)

##we will run the environment here and return the new set of frames
class getframes:
    skip = 4 #The number of frames we skip
    ct = 0
    def __init__(self,imsize):
        self.imsize = imsize

    def rgb2gray(self, image):
        return np.dot(image[...,:3], [0.299, 0.587, 0.114])

    def getcurrstack(self, env, action, prevstack):
        observation, reward, done, info =env.step(action)
        #strs = './videos/lqn/'+'sample' + str(self.ct) + '.jpg'
        #imsave(strs, observation)
        #print(prevstack[:,:,0].shape)
        newobs = self.preprocess(observation)
        #print(newobs)
        newobs = np.reshape(newobs, (self.imsize[0], self.imsize[1], 1))
        #print(newobs)
        newobs = np.append(newobs, prevstack[:,:,0:3], axis=2) #doubt2, is appending correct?
        return newobs, reward, done   

    def preprocess(self, img):
        img = self.rgb2gray(img)
        #result = result.convert('L')
        result = imresize(img,(self.imsize[0],self.imsize[1])).astype(np.float32)
        result = result/np.max(result)
        self.ct =self.ct + 1
        #print('In the pre process stage')
        #strs = '../images'+'sample'+str(self.ct)+'.png'
        #imsave(strs,result)
        
        return result
        
##We will create the replaymemory here
class ReplayMemory:

    maxsize=25000
    def __init__(self):
        self.mem = deque()

    def append(self, state, action, reward, terminal, nextstate):
        self.mem.append((state, action, reward, terminal, nextstate))
        if len(self.mem) > self.maxsize:
            self.mem.popleft()

    def sample(self, batch_size):
        if len(self.mem) < batch_size:
            print('The batch size you have requested is more than the replay memory')
            return
        return random.sample(self.mem, batch_size)


def main():  # noqa: D103
    f = open("dualdqn.txt","w")
    parser = argparse.ArgumentParser(description='Run DQN on Atari Breakout')
    parser.add_argument('--env', default='Breakout-v0', help='Atari env name')
    parser.add_argument(
        '-o', '--output', default='atari-v0', help='Directory to save data to')
    parser.add_argument('--seed', default=0, type=int, help='Random seed')

    args = parser.parse_args()
    #args.input_shape = tuple(args.input_shape)

    args.output = '1'#get_output_folder(args.output, args.env)

    # here is where you should start up a session,
    # create your DQN agent, create your model, etc.
    # then you can run your fit method.
    ##########defining some hyper parameters here####################
    gamma = 0.99
    batch_size = 32
    skip = 4
    minsamples = 10000 #minimum observations in our replay memory
    lr = 0.0001
    loadfreq = 10000
    #################################################################
    env = gym.make('SpaceInvaders-v0')
    #env = gym.make('Breakout-v0')
    ind = 1
    #env = wrappers.Monitor(env, './videos/lqn/space-invadors-v1-' +str(ind), force=True)
    #env2.reset()
    #env = gym.make('CartPole-v0')
    initstate = env.reset()
    primmodel = create_model(4,(84,84),env.action_space.n)
    targetmodel = create_model(4,(84,84),env.action_space.n)
    mem = ReplayMemory() 
    print('The model created is')
    print(primmodel.summary())

    #********Getting the frames from the gym environment now***********#
    
    #prevstack = preprocess(initstate, (84,84))
   
    x = tf.placeholder(tf.float32, shape=(None,84,84,4))
    a = tf.placeholder(tf.float32, [None, env.action_space.n])
    y = tf.placeholder(tf.float32, [None])
    targx = tf.placeholder(tf.float32, shape=(None,84,84,4)) 
    out = primmodel(x)
    targout = targetmodel(targx)
    targetq = tf.reduce_sum(tf.multiply(out, a),reduction_indices = 1)
    absloss = tf.reduce_mean((y-targetq))
    mseloss = tf.reduce_mean(tf.square(y-targetq))
    max_grad = 1
    huberloss = tf.where(tf.abs(absloss) < max_grad, 0.5 * tf.square(absloss), max_grad * (
        tf.abs(absloss) - max_grad * 0.5))
    trainer = tf.train.AdamOptimizer(lr).minimize(huberloss)
    prevstack = np.zeros((84,84,4),dtype=np.float32)
    imsize = (84,84)
    frame = getframes(imsize)    
    pol = policy() #Initializing the policy class here
    maxepisodes = 1
    episode = 0
    saveiter = 100000
    numsave = 0
    #with tf.Session() as sess:
    for ra in range(0,maxepisodes):
        prevstack = np.zeros((84,84,4),dtype=np.float32)
        sess = tf.InteractiveSession()
        #tf.global_variables_initializer().run()
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        counts = 0
        print('In the episode', episode)
        episode = episode + 1
        accreward = 0
        if episode > 1:
            minsamples = 0
        iterations = minsamples + 4980000
        printiter = 200
        
        #for i in range(0,skip): don't know what exactly to skip
        prevstack = np.zeros((84,84,4),dtype=np.float32)
        while iterations>0:
            if iterations % loadfreq == 0:
                print('Loaded weights to the online network')
                targetmodel.set_weights(primmodel.get_weights()) 
            if iterations % saveiter ==0:
                numsave = numsave+1
                strs = 'dualdqn/' + str(numsave) + '.h5'
                primmodel.save(strs)
            for j in range(0,minsamples): #will run once only
                prevstack = np.reshape(prevstack,(1,imsize[0],imsize[1],4))
                outs = sess.run(out, feed_dict={x:prevstack})
                action = pol.greedyepsilon(1,outs)
                prevstack = np.reshape(prevstack,(84,84,4))
                currstack,reward,done = frame.getcurrstack(env, action, prevstack)
                #accreward = accreward + reward
                if done:
                    env.reset()
                mem.append(prevstack, action, reward, done, currstack)
                prevstack = np.copy(currstack)
                if j==minsamples-1:
                    prevstack = np.zeros((84,84,4),dtype=np.float32)
                iterations = iterations - 1
            #We are done with the replay memory
            minsamples = 0
            #prevstack = np.zeros((84,84,4),dtype=np.int) #doubt3: making it zero again at each iteration? shouldn't have existed
            prevstack = np.reshape(prevstack,(1,imsize[0],imsize[1],4))
            outs = sess.run(out, feed_dict={x:prevstack})
            strs = 'Iteration is' + str(iterations) + '\n'
            f.write(strs)
            strs = 'The mac q values is' + str(np.max(outs))+ '\n'
            f.write(strs)
            strs = 'The min q value is '+ str(np.min(outs)) + '\n'
            f.write(strs)
            strs = 'The average of q value is'+ str(np.average(outs))+ '\n'
            f.write(strs) 
            #print('The maximum q value is %d and the iteration is %d',np.max(outs), counts)
            action = pol.ldgreedyepsilon(outs)
            #print('The action chosen is', action)
            prevstack = np.reshape(prevstack,(84,84,4))
            currstack, reward, done = frame.getcurrstack(env, action, prevstack)
            #print(np.max(currstack[:,:,0]-prevstack[:,:,0]),np.min(currstack[:,:,0]-prevstack[:,:,0]))
            #print(np.max(currstack[:,:,1]-prevstack[:,:,0]),np.min(currstack[:,:,1]-prevstack[:,:,0]))
            #print(np.max(currstack[:,:,2]-prevstack[:,:,1]),np.min(currstack[:,:,2]-prevstack[:,:,1]))
            #print(np.max(currstack[:,:,3]-prevstack[:,:,2]),np.min(currstack[:,:,3]-prevstack[:,:,2]))
            if done:
                env.reset()
                strs = 'Terminal state reached, reward accumulated is' + str(accreward) + '\n'
                f.write(strs)
                accreward =0 
            mem.append(prevstack, action, reward, done, currstack)
            prevstack = np.copy(currstack) 
            currstack = np.reshape(currstack,(1,imsize[0],imsize[1],4))
            #print('The reward is', reward) 
            accreward = accreward + reward
            sample = random.sample(mem.mem, batch_size)
            prevsample = [ind[0] for ind in sample]
            actionsample = [ind[1] for ind in sample]
            rewardsample = [ind[2] for ind in sample] 
            for k in range(0,len(rewardsample)):
                if rewardsample[k] > 0:
                    rewardsample[k] = 1
                if rewardsample[k] < 0:
                    rewardsample[k] = -1
            termsample = [ind[3] for ind in sample]
            currsample = [ind[4] for ind in sample]
            
            #state selection by online network only
            qstate = sess.run(out, feed_dict={x: currsample})
            #Passed through the target network
            qvalues = sess.run(targout, feed_dict={targx: currsample})
            targets = []
            sumq = 0
            for m in range(0,batch_size):
                if termsample[m]:
                    targets.append(rewardsample[m])
                else:
                    statemax = np.argmax(qstate[m]) #return indices of this max
                    targets.append(rewardsample[m] + gamma*(qvalues[m][statemax]))

            actionnames = np.zeros((len(actionsample),env.action_space.n), dtype=np.int)
            for i in range(0, len(actionsample)):
                actionnames[i][actionsample[i]] = 1

            #print('action names is',actionnames[0])
            _,lossval = sess.run([trainer, mseloss],feed_dict ={y: targets, a:actionnames, x:prevsample}) #doubt4: what all will get updated ?? both the weights?
            #print(targets)
            #if iterations % printiter ==0:
            strs = 'The loss is' + str(lossval)+'\n'
            f.write(strs) 
            iterations = iterations - 1
            counts = counts + 1
            
            strs = 'The accumulated reward is ' + str(accreward) + '\n'
            f.write(strs)
    #os.system('ffmpeg -framerate 25 -i ./videos/lqn/sample%d.jpg episode.mp4')
    f.close()

if __name__ == '__main__':
    main()
