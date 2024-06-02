import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random
import matplotlib.pyplot as plt
import json
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
    def __init__(self, state_size, action_size, seed, lr=0.0001, gamma=0.99, tau=0.01):
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

        # Discretize the action space
        self.action_space = np.linspace(-2.0, 2.0, action_size)

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
            action_index = np.argmax(action_values.cpu().data.numpy())
        else:
            action_index = random.choice(np.arange(self.action_size))

        action = self.action_space[action_index]
        return action, action_index

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

def train_dqn(seed, lr=0.0001, gamma=0.99, tau=0.01, n_episodes=1000, max_t=200, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    env = gym.make('Pendulum-v1')
    state_size = env.observation_space.shape[0]
    action_size = 21  # Number of discrete actions
    agent = DQNAgent(state_size, action_size, seed, lr, gamma, tau)

    eps = eps_start
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract the actual state from the tuple
        total_reward = 0
        for t in range(max_t):
            action, action_index = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step([action])
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the actual next state from the tuple
            done = terminated or truncated
            agent.step(state, action_index, reward, next_state, done)
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

# Training with different seeds
seeds = [0, 1, 2]
results = {}

if not os.path.exists('models'):
    os.makedirs('models')

for seed in seeds:
    print(f"Training with seed={seed}")
    scores, model_state = train_dqn(seed)
    key = f"seed_{seed}"
    results[key] = scores

    # Save the model
    model_path = f'models/dqn_pendulum_{key}.pth'
    torch.save(model_state, model_path)

    # Save the scores
    with open(f'scores_pendulum_{key}.json', 'w') as f:
        json.dump(scores, f)

# Plotting the results
plt.figure(figsize=(12, 8))
for key, scores in results.items():
    plt.plot(scores, label=key)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN Training Performance on Pendulum-v1 with Different Seeds')
plt.legend()
plt.show()
