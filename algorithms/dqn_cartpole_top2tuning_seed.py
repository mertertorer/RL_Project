import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import json
import matplotlib.pyplot as plt
import os

# Define the Q-network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, experience):
        self.memory.append(experience)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)

# DQN Agent
class DQNAgent:
    def __init__(self, state_size, action_size, seed, lr=0.001, gamma=0.99, tau=1e-3):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=lr)

        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        self.gamma = gamma
        self.tau = tau

    def step(self, state, action, reward, next_state, done):
        self.memory.push((state, action, reward, next_state, done))
        
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            self.learn(experiences)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = zip(*experiences)

        states = torch.from_numpy(np.vstack(states)).float()
        actions = torch.from_numpy(np.vstack(actions)).long()
        rewards = torch.from_numpy(np.vstack(rewards)).float()
        next_states = torch.from_numpy(np.vstack(next_states)).float()
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float()

        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        Q_targets = rewards + (self.gamma * Q_targets_next * (1 - dones))
        
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        
        loss = nn.MSELoss()(Q_expected, Q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0-self.tau)*target_param.data)

def train_dqn(lr, gamma, tau, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995, seed=0):
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size, seed, lr, gamma, tau)

    eps = eps_start
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract the actual state from the tuple
        total_reward = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, info = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the actual next state from the tuple
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        scores.append(total_reward)
        eps = max(eps_end, eps_decay*eps)
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores[-100:])}")

    env.close()

    return scores, agent.qnetwork_local.state_dict()

# Top 2 hyperparameter configurations
top_configs = [
    {'lr': 0.0005, 'gamma': 0.99, 'tau': 0.01},
    {'lr': 0.0001, 'gamma': 0.99, 'tau': 0.01}
]

# Seeds
seeds = [0, 1, 2]

results = {}
if not os.path.exists('models'):
    os.makedirs('models')

for config in top_configs:
    for seed in seeds:
        lr = config['lr']
        gamma = config['gamma']
        tau = config['tau']
        print(f"Training with lr={lr}, gamma={gamma}, tau={tau}, seed={seed}")
        scores, model_state = train_dqn(lr, gamma, tau, seed=seed)
        key = f"lr_{lr}_gamma_{gamma}_tau_{tau}_seed_{seed}"
        results[key] = scores

        # Save the model
        model_path = f'models/dqn_cartpole_{key}.pth'
        torch.save(model_state, model_path)

        # Save the scores
        with open(f'scores_{key}.json', 'w') as f:
            json.dump(scores, f)

# Plotting the results for each configuration with different seeds
for config in top_configs:
    plt.figure(figsize=(12, 8))
    for seed in seeds:
        key = f"lr_{config['lr']}_gamma_{config['gamma']}_tau_{config['tau']}_seed_{seed}"
        plt.plot(results[key], label=f"seed_{seed}")
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f"DQN Training Performance on CartPole-v1 with lr={config['lr']}, gamma={config['gamma']}, tau={config['tau']}")
    plt.legend()
    plt.show()
