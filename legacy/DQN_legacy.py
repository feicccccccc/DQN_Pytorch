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
TD(0) to update Q(s,a)

General idea:
Learn and generalise the Q(s,a) using DNN. Base on the Q(s,a), act ε-greedy

important technique:
loss function (between true Q and estimated Q'): Mean square error
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


class DQN(nn.Module):
    """
    A class to represent the function approximator for Q(s,a)
    The input will be a array of 4 frame such that it capture the change of the system,
    and make it as part of the state.
    i.e. a state of a particle is characterise by (s,v) position and velocity
    """

    def __init__(self, h, w, stack, outputs):
        """
        The layer structure of CNN
        :param h: height of the input image
        :param w: width of the input image
        :param outputs: number of output of the (discrete) action space
        :return: probability of choosing action in the (discrete) action space
        """
        super(DQN, self).__init__()
        # input channel: 4 stack frame, 3 RGB channel
        # output channel: 16
        self.conv1 = nn.Conv2d(stack * 3, 16, kernel_size=5, stride=2)
        self.conv1_drop = nn.Dropout2d(p=0.5)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv1_relu = nn.ReLU()

        # input channel: 16
        # output channel: 32
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv2_relu = nn.ReLU()

        # input channel: 32
        # output channel: 64
        self.conv3 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.conv3_drop = nn.Dropout2d(p=0.5)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv3_relu = nn.ReLU()

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        # output size = (input - kernel_size + 2 * Padding) / stride + 1
        def conv2d_size_out(size, kernel_size=5, stride=2):
            # Floor division
            return int(((size - kernel_size) // stride) + 1)

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))

        # output * number of channel
        linear_input_size = convw * convh * 64

        self.fc1 = nn.Linear(linear_input_size, outputs)
        self.head = nn.Softmax(dim=-1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        """
        Forward with the layer define in the attribute
        :param x: Input Image stack
        :return:
        """
        x = self.conv1(x)
        x = self.conv1_drop(x)
        x = self.bn1(x)
        x = self.conv1_relu(x)

        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.bn2(x)
        x = self.conv2_relu(x)

        x = self.conv3(x)
        x = self.conv3_drop(x)
        x = self.bn3(x)
        x = self.conv3_relu(x)

        x = x.flatten()
        x = self.fc1(x)
        x = self.head(x)

        return x
