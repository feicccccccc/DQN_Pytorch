"""
Exercise for Implementing DQN
following:
Pytorch Documentation: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
DeepMind Youtube lecture: https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ
Original Paper: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf

Some code is being copied from the Pytorch official tutorial with slight modification to make thing more familiar to me.
Including:
1. Slight change of the NN structure, add dropout and softmax as output
2. Track the whole image instead of only the cart
3. No specific transform done to the image, not even normalisation
4. Use Adam optimiser

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
forward view TD(0) to update Q(s,a) (1 step look ahead)

General idea:
Learn and generalise the Q(s,a) using DNN. Base of the Q(s,a), act ε-greedy

important technique:
Eligibility trace E(θ): contribution to the Q by parameter θ
loss function (between true Q and estimated Q'): Mean square error
Experience replay: Reduce correlation between path to reduce variance

"""

import gym
import numpy as np
import matplotlib.pyplot as plt
import random
import math
from collections import namedtuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from DQN import ReplayMemory, DQN, Transition

from itertools import count

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

NUMBER_OF_FRAME = 4

# Get number of actions from gym action space
n_actions = env.action_space.n

# nn.conv2d input format (channel, height, width)
# on
init_screen = env.render(mode='rgb_array')  # dim: (800, 1200, 3)
init_screen = np.expand_dims(init_screen.transpose((2, 0, 1)), axis=0)
init_screen = np.repeat(init_screen, NUMBER_OF_FRAME, axis=0)
_, _, screen_height, screen_width = init_screen.shape  # shape: (frame, channel, height, width)

# Q-learning is off-policy

behavioural_net = DQN(screen_height, screen_width, NUMBER_OF_FRAME, n_actions).to(device)
target_net = DQN(screen_height, screen_width, NUMBER_OF_FRAME, n_actions).to(device)

# Make sure they have the same initial weight
target_net.load_state_dict(behavioural_net.state_dict())
# Set net to evaluation mode
target_net.eval()

# optimise using Adam
optimizer = optim.Adam(behavioural_net.parameters())
memory = ReplayMemory(10000)

steps_done = 0  # global variable to remember how many step have been taken


def select_action(state):
    """
    ε-greedy Action
    :param state: state s
    :return: policy a
    """
    global steps_done
    # Return the next random floating point number in the range [0.0, 1.0).
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # in short, find the largest column in the network output, which is the predicated largest Q(s,a)
            action_out = behavioural_net(state)
            print("Step: {}, eps_threshold: {}, network output: {}".format(steps_done, eps_threshold, action_out))
            action_out = action_out.max(1)[1].view(1, 1)
            return action_out
    else:
        action_out = torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)
        print("Step: {}, eps_threshold: {},  pick random action {}".format(steps_done, eps_threshold, action_out))
        return action_out


episode_durations = []


def plot_durations():
    """
    Helper function to plot from the global array episode_durations
    :return: None
    """
    plt.figure(2)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated


# Training loop

def optimize_model():
    # Not enough experience / memory
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch
    # zip return an iterator of tuple for the array in the argument
    # *expression
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = behavioural_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in behavioural_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == '__main__':
    num_episodes = 50
    state = init_screen
    for i_episode in range(num_episodes):

        # Current state is the current screen + 3 frame before
        # TODO: Try RNN here?
        env.reset()
        last_screen = np.expand_dims(env.render(mode='rgb_array').transpose((2, 0, 1)), axis=0)
        state = np.append(state[0:NUMBER_OF_FRAME-1], last_screen, axis=0)
        state = torch.from_numpy(state)

        # itertools count using natural number
        for t in count():
            # Select and perform an action
            action = select_action(state)
            _, reward, done, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)

            # S -> A -> R

            # Observe new state
            last_screen = np.expand_dims(env.render(mode='rgb_array').transpose((2, 0, 1)), axis=0)
            state = np.append(state[0:NUMBER_OF_FRAME - 1], last_screen, axis=0)
            state = torch.from_numpy(state)
            if not done:
                next_state = state
            else:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            # Won't do without enough experience
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        # similar to police evaulation
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(behavioural_net.state_dict())

    print('Complete')
    env.render()
    env.close()
    plt.ioff()
    plt.show()
