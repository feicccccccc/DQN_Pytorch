"""
Exercise for Implementing DQN
following:
Pytorch Documentation: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
DeepMind Youtube lecture: https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
Original Paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

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

plt.ion()  # turn on interactive mode

env = gym.make("CartPole-v0").unwrapped  # getting access to more detail information for the env
observation = env.reset()

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    A class to stored the trajectory
    """

    def __init__(self, capacity):
        self.capacity = capacity  # size of buffer
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # cyclic buffer http://www.mathcs.emory.edu/~cheung/Courses/171/Syllabus/8-List/array-queue2.html
        # First in first out
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
    Return the a random sample from the memory
    :param batch_size: how many to return
    :return: number of batch_size random sample from the memory
    """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
