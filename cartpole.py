"""
Exercise for Implementing DQN
following:
Pytorch Documentation: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
DeepMind Youtube lecture: https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
Original Paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Some code is being copied from the Pytorch modification with slight modification to make thing more familiar to me.
Including:
1. Slight change of the NN structure, add dropout and softmax as output
2. Track the whole image instead of only the cart
3. No specific transform done to the image, not even normalisation

Note to the DQN and MDP Process:
Under MDP framework, every state s have a associate Action Value Function Q(s,a), which
denote the
1. Discounted Sum of Reward
2. In state s and performed action a
3. Following Policy π(a|s) (define by a probability distribution of action a in state s)

Estimation of the Q(s,a) : DNN with input = observation and state
behavioural Policy : ε-greedy
target Policy: greedy

Update rule:
forward view TD(λ) to update Q(s,a)

General idea:
Learn and generalise the Q(s,a) using DNN. Base of the Q(s,a), act ε-greedy

important technique:
Eligibility trace E(θ): contribution to the Q by parameter θ
loss function (between true Q and estimated Q': Mean square error
Experience replay: Reduce correlation between path to reduce variance

"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from DQN import ReplayMemory, DQN

plt.ion()  # turn on interactive mode

# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make("CartPole-v0").unwrapped  # getting access to more detail information for the env
observation = env.reset()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

BATCH_SIZE = 128
# discount factor
GAMMA = 0.999
# decaying ε-greedy
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200

TARGET_UPDATE = 10

# Get number of actions from gym action space
n_actions = env.action_space.n

# nn.conv2d input format (channel, height, width)
# on
init_screen = env.render(mode='rgb_array') # dim: (800, 1200, 3)
init_screen = np.repeat(init_screen.transpose((2, 0, 1)), 4, axis=0)



print(n_actions)