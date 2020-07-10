# Multi-agent continuous control with MADDPG 

## Introduction
Implementation of MADDPG algorithm to solve the multi-agent contiunous action space of Unity Tennis Environment :tennis:.
Random Agent            |  Trained Agent
:----------------------:|:-------------------------:
![](https://github.com/Zaharah/DeepRL-MADDPG-Tennis-Unity/blob/master/raw_CnC.gif)  |  ![](https://github.com/Zaharah/DeepRL-MADDPG-Tennis-Unity/blob/master/sloved_CnC.gif)

 - *Multiple Agents*: Two agents playing Tennis with each other. Each agent is trying compete and collaborate to hit the ball with the racket.  
 - *Environment*: In the Tennis environment (:tennis:), two agents control rackets to bounce a ball over a net. 
 - *Action size*: Each agent receives a vector of actions consisting of `2` values, corresponding to to movement towards or away from net and jumping. A single action can take any value between `-1` to `1`. 
 - *Observation space*: The state space of each agent consists of `8` variables consisting of position of ball, its velocity among other details. 
 - *Rewards*: There is `+1` reward(:+1:) for hitting the ball over the net, the reward of `-1` is given in case ball hit the ground or agent hit the out of the bound ball. 
 - *Goal*: The environment is considered solved (:star:) when an agent accumulates an average reward of `+0.5` for `100` consecutive episodes after taking the maximum over both agents. 
 - *Apporach*: A MADDPG algorithms is implemented to solve the Tennis environment having multi-agent continous actions space.
 
  ## Getting Started
1. Follow the instructions on the Udacity Repo [Here](https://github.com/udacity/deep-reinforcement-learning/#dependencies)
2. Download the /Unity Tennis environment. Select the environment that matches your operating system:
  - Linux: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
  - Mac OSX: [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
  - Windows (32-bit): [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
  - Windows (64-bit): [Click Here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)
3. Make sure that the downloaded environment and these code files are in same the folder. 
 
## Instructions

 1. To explore the code, open Python Notebook `Tennis-CnC.ipynb`.
 2. To see an `agent in action`, skip the training action and load the trained weights.   
