import sys
import argparse
import logging
import torch
import matplotlib.pyplot as plt
from models import DDPG
from utils import *
import environment

parser = argparse.ArgumentParser()
parser.add_argument('--seed', default=0, type=int, help='Random Seed(default: 0)')
parser.add_argument('--env', default='environment.py', help='the environment where the agents to be trained')
parser.add_argument('--num_episode', default=20, type=int, help='Number of episode')
parser.add_argument('--num_observation', default=10000, type=int, help='Number of observation/timestep')
parser.add_argument('--batch_size', default=64, type=int, help='Number of batch size')
parser.add_argument('--gamma', default=0.99, help='Discount factor')
parser.add_argument('--tau', default=0.01, help='Soft update factor in target network')
parser.add_argument('--actor_lr', default=1e-4, help='Actor network learning rate')
parser.add_argument('--critic_lr', default=1e-3, help='Critic network learning rate')
parser.add_argument('--actor_hidden_size', default=100, help='Number of unit in Actor network hidden layers')
parser.add_argument('--critic_hidden_size', default=100, help='Number of unit in Critic network hidden layers')
parser.add_argument('--gh_hidden_size', default=500, help='Number of unit in graph hidden layers')
parser.add_argument('--gh_depth', default=10, help='Graph network input channels')
parser.add_argument('--gh_growth_rate', default=12, help='Number of graph networks output channels')
parser.add_argument('--gh_reduction', default=0.5, help='Reduction factor in GDB block')
parser.add_argument('--gh_num_head', default=8, help='Number of attention head')
args = parser.parse_args()




device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

env = environment.Environ()
agent = DDPG(env)
noise = OUNoise(env.action_space)

num_episode = 20
num_obs = 10000
batch_size = 64
gamma = 0.99
rewards = []
avg_rewards = []
loss = []


for episode in range(num_episode):
    env.random_game()   # need modification
    noise.reset()   # need modification
    episode_reward = 0

    for obs_step in range(num_obs):
        action = agent.get_action(state)
        action = noise.get_action(action, obs_step)
        next_state = env.caching_strategy() # need modification
        next_state, reward = env.step(action)  # step no longer exists
        agent.memory.push(state, action, reward, next_state)

        if len(agent.memory) > batch_size:   # ReplayMemory need modification
            agent.update(batch_size)
            # policy_loss, critic_loss = agent.update(batch_size)
        state = next_state
        episode_reward += reward

        sys.stdout.write("episode: {}, reward: {}, average_reward: {} \n".format(episode, np.round(episode_reward, decimals=6), np.mean(rewards[-10:])))

        rewards.append(episode_reward)
        avg_rewards.append(np.mean(rewards[-10:]))   # clip and return last 10 entries in the list

torch.save({'actor_state_dict': agent.actor.state_dict(), 'critic_state_dict': agent.critic.state_dict()}, 'DDPG.pth')

plt.plot(rewards)
plt.plot(avg_rewards)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.show()


'''
for episode_idx in range(1, num_episodes + 1):
    epsilon = epsilon_by_episode(episode_idx)
    action = model.act(state, epsilon)   # need modification

    next_state, reward = env.step(action)   # need modification
    replaymemory.push(state, action, reward, next_state)

    state = next_state
    episode_reward += reward
    all_rewards.append(episode_reward)

    if len(replaymemory) > batch_size:
        loss = compute_loss(batch_size, gamma)
        losses.append(loss.data[0])

    if episode_idx % 50 == 0:
        plt.plot(episode_idx, all_rewards)
        plt.plot(episode_idx, losses)
'''

