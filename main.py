"""
Reference and credit:
https://www.youtube.com/watch?v=UlJzzLYgYoE

Main loop
Slightly change to the network layer structure and parameter naming
Change gym wrapper and observation
Add a lot of comment for reference
"""

import gym
import numpy as np
from agent import DQNAgent
from utils import plot_learning_curve, make_env

NUMBER_OF_FRAME = 4

if __name__ == '__main__':

    env = make_env("CartPole-v0")
    init_screen = env.reset()
    best_score = -np.inf
    load_checkpoint = False  # if user want to restart from checkpoint
    n_games = 1000  # number of episode

    # replace target network with evaluation network after 1000 step
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.001,
                     input_dims=init_screen.shape,
                     n_actions=env.action_space.n, mem_size=50000, eps_min=0.2,
                     batch_size=64, replace=200, eps_dec=5e-4,
                     checkpoint_dir='models/', algo='DQNAgent',
                     env_name='CartPole-v0-RGB')

    if load_checkpoint:
        agent.load_models()

    # For record
    n_steps = 0
    scores, eps_history, steps_array = [], [], []

    for i in range(n_games):
        done = False
        observation = env.reset()

        score = 0
        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward

            if not load_checkpoint:
                # store the transition (s,a,r,s') inside the replay memory
                agent.store_transition(observation, action, reward, next_observation, int(done))
                # learn through the experience (if there's enough batches)
                agent.learn()
            observation = next_observation
            n_steps += 1

        # After 1 episode finish
        # keep record stuff
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i, 'score: ', score,
              ' average score %.1f' % avg_score, 'best score %.2f' % best_score,
              'epsilon %.2f' % agent.epsilon, 'steps', n_steps)

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        if load_checkpoint and n_steps >= 18000:
            break

    # for graph piloting
    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    x = [i + 1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)
