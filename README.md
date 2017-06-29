# DeepRLAtari
Implementation of popular RL algorithms for Atari games

In LQN folder the following are implemented
Linear.py --> a simple linear network without any replay memory or target fixing
LinearRepMem.py --> a linear network with replay memory and target fixing
LinearDouble --> a double linear network with replay memory and target fixing

In DQN folder the following are implemented
DQN.py --> A Deep Q Network 
DoubleDQN.py --> A Double Deep Q Network
DuellingDQN --> A DQN with Duelling architecture

In general DQN's outperform LQN's with the Linear.py having the worst performance. I have found duelling DQN to give the best performance but it might vary.

Also, an indication that your function is learning is to look at the Q values, which must have an increasing trend and also the avg rewards which have to go up.

The above networks were trained for 5 million iterations.
