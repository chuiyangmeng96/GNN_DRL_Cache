import matplotlib.pyplot as plt
import torch
from models import Actor
import environment
obs_step = 200
# N = compute(obs_step)
obs = list(range(obs_step))
env = environment.Environ()
checkpoint = torch.load('DDPG.pth')
agent = Actor()
agent.load_state_dict(checkpoint['actor_state_dict'])   # we only need actor network to calculate action at this point

for i in range(5):
    state = env.random_game()
    total_reward = 0
    rewards = []
    for j in range(obs_step):
        action = agent.get_action(state)
        state, reward = env.step(action)
        total_reward += reward
        rewards.append(total_reward)
    fig = plt.figure()
    plt.plot(obs, rewards)
    plt.xlabel('Time Step (s)')
    plt.ylabel('Cumulative Reward')
    plt.title('Test Cumulative Reward vs Time Step in DDPG method')
    plt.show()
