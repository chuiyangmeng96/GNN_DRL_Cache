import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
from utils import *
from graph_models import GraphDensenet


class Actor(nn.Module):
    def __init__(self, s_dim, hidden_size, a_dim):
        super(Actor, self).__init__()

        self.fc1 = nn.Linear(s_dim, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_size, a_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.relu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        actions_value = x   # need modification
        return actions_value


class Critic(nn.Module):
    def __init__(self, s_dim, hidden_size, a_dim):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(s_dim + a_dim, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_size, a_dim)
        self.fc3.weight.data.normal_(0, 0.1)
        self.relu = nn.LeakyReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        q_value = x   # need modification
        return q_value

class MACritic(nn.Module):
    def __init__(self, num_agent, s_dim, hidden_size, a_dim):
        super(MACritic, self).__init__()
        obs_dim = s_dim * num_agent
        act_dim = a_dim * num_agent
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(hidden_size + act_dim, hidden_size)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.fc3.weight.data.normal_(0, 0.1)
        self.relu = nn.LeakyReLU()

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        q_value = x


        return q_value


class DDPG(nn.Module):
    def __init__(self, gh_input_size, gh_hidden_size, gh_output_size, gh_depth, gh_growth_rate, gh_reduction, gh_bottleneck,
                 gh_dropRate, gh_num_head, gh_slope_alpha, gh_bias,
                 actor_hidden_size = 100, critic_hidden_size = 100, gamma = 0.99, tau = 0.01,
                 actor_learning_rate = 1e-4, critic_learning_rate = 1e-3,
                 max_memory_size = 50000):
        super(DDPG, self).__init__()
        # self.num_states =
        # self.num_states = env.observation_space.shape[1]   # need modification
        # self.num_actions = env.action_space.shape[1]   # need modification
        self.gamma = gamma
        self.tau = tau

        # Network setup
        self.GraphDensenet = GraphDensenet(gh_input_size, gh_hidden_size, gh_output_size, gh_depth, gh_growth_rate, gh_reduction, gh_bottleneck,
                 gh_dropRate, gh_num_head, gh_slope_alpha, gh_bias) # need revision
        self.actor = Actor(self.num_states, actor_hidden_size, self.num_actions)
        self.actor_target = Actor(self.num_states, actor_hidden_size, self.num_actions)
        self.critic = Critic(self.num_states + self.num_actions, critic_hidden_size, self.num_actions)
        self.critic_target = Critic(self.num_states + self.num_actions, critic_hidden_size, self.num_actions)
        #   need modification

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

        # training stage
        self.memory = ReplayMemory(max_memory_size)
        self.critic_criterion = nn.MSELoss()
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_learning_rate)

    def get_action(self, state):
        state = Variable(torch.from_numpy(state).float().unsqueeze(0))
        action = self.actor.forward(state)
        action = action.detach().numpy()[0, 0]
        action =  # need revision
        return action

    def update(self, batch_size):
        state, action, reward, next_state = self.memory.sample(batch_size)
        state = torch.FloatTensor(state)
        action = torch.FloatTensor(action)
        reward = torch.FloatTensor(reward)
        next_state = torch.FloatTensor(next_state)

        # Critic loss
        q_value = self.critic.forward(state, action)
        next_actions = self.actor_target.forward(next_state)
        next_q_value = self.critic_target.forward(next_state, next_actions.detach())
        expected_q_value = reward + self.gamma * next_q_value
        critic_loss = self.critic_criterion(q_value, expected_q_value)

        # Actor loss provided by Critic network
        policy_loss = -self.critic.forward(state, self.actor.forward(state)).mean()

        # gradient descent in actor/critic network
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1 - self.tau))

        return policy_loss, critic_loss

class MADDPG(nn.Module):
    def __init__(self, gh_input_size, gh_hidden_size, gh_output_size, gh_depth, gh_growth_rate, gh_reduction,
                 gh_bottleneck,
                 gh_dropRate, gh_num_head, gh_slope_alpha, gh_bias,
                 actor_hidden_size=100, critic_hidden_size=100, gamma=0.99, tau=0.01,
                 actor_learning_rate=1e-4, critic_learning_rate=1e-3,
                 max_memory_size=50000):
        super(MADDPG, self).__init__()



'''
def compute_loss(batch_size, gamma):
    state, action, reward, next_state = replaymemory.sample(batch_size)
    state = Variable(torch.FloatTensor(np.float32(state)))
    next_state = Variable(torch.FloatTensor(np.float32(next_state)))
    action = Variable(torch.LongTensor(action))
    reward = Variable(torch.FloatTensor(reward))

    q_values = model(state)
    next_q_values = model(next_state)
    q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
    next_q_value = next_q_values.max(1)[0]
    expected_q_value = reward + gamma * next_q_value  # need modification

    loss = (q_value - expected_q_value.detach()).pow(2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


    # delete if unnecessary
    def act(self, state, epsilon):
        if random.random() > epsilon:
            state = Variable(torch.FloatTensor(state).unsqueeze(0))
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(env.action_space.n)  # need modification
        return action


# model = DDPG()  # need modification1

'''
'''

def epsilon_by_episode(num, epsilon_start, epsilon_final, epsilon_decay):  # epsilon decay function
    return epsilon_final + (epsilon_start - epsilon_final) * math.exp(-1. * num / epsilon_decay)


epsilon_start = 1.0
epsilon_final = 0.01
epsilon_decay =  500'''