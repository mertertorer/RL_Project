import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import matplotlib.pyplot as plt
import cv2

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
    def __init__(self, state_size, action_size, seed, discrete=True):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.discrete = discrete

        self.qnetwork_local = QNetwork(state_size, action_size)
        self.qnetwork_target = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=0.001)

        self.memory = ReplayMemory(10000)
        self.batch_size = 64
        self.gamma = 0.99
        self.tau = 1e-3

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
            action = np.argmax(action_values.cpu().data.numpy())
        else:
            action = random.choice(np.arange(self.action_size))

        if not self.discrete:
            action = np.array([action])
        return action

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

def create_env(env_name):
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n if isinstance(env.action_space, gym.spaces.Discrete) else env.action_space.shape[0]
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    return env, state_size, action_size, discrete

def initialize_agent(state_size, action_size, discrete):
    return DQNAgent(state_size, action_size, seed=0, discrete=discrete)

def train_agent(env_name, model_path, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    # Create the environment
    env, state_size, action_size, discrete = create_env(env_name)
    
    # Initialize the agent
    agent = initialize_agent(state_size, action_size, discrete)
    
    # Training loop
    eps = eps_start
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract the actual state from the tuple
        total_reward = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the actual next state from the tuple
            done = terminated or truncated
            agent.step(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break
        scores.append(total_reward)
        eps = max(eps_end, eps_decay * eps)
        if i_episode % 100 == 0:
            print(f"Episode {i_episode}\tAverage Score: {np.mean(scores[-100:])}")
    
    # Save the trained model
    torch.save(agent.qnetwork_local.state_dict(), model_path)
    
    # Plotting the scores
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'DQN Training Performance on {env_name}')
    plt.show()
    
    env.close()

def visualize_agent(env, agent, model_path, n_episodes=5):
    agent.qnetwork_local.load_state_dict(torch.load(model_path))
    agent.qnetwork_local.eval()  # Set the network to evaluation mode
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract the actual state from the tuple
        total_reward = 0
        done = False
        frames = []
        while not done:
            frame = env.render(mode='rgb_array')
            frames.append(frame)
            action = agent.act(state, eps=0.0)  # Use a greedy policy (no exploration)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the actual next state from the tuple
            done = terminated or truncated
            state = next_state
            total_reward += reward
        print(f"Episode {i_episode}\tTotal Reward: {total_reward}")
        
        # Save the video
        height, width, layers = frames[0].shape
        video = cv2.VideoWriter(f'episode_{i_episode}.avi', cv2.VideoWriter_fourcc(*'DIVX'), 30, (width, height))
        for frame in frames:
            video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        video.release()
    
    env.close()

def main():
    # Example to train the agent on CartPole
    # train_agent('CartPole-v1', 'dqn_cartpole.pth')

    # Example to train the agent on Pendulum
    train_agent('Pendulum-v1', 'dqn_pendulum.pth')

    # Create the environment
    env, state_size, action_size, discrete = create_env('Pendulum-v1')
    
    # Initialize the agent
    agent = initialize_agent(state_size, action_size, discrete)
    
    # Visualize the trained agent
    visualize_agent(env, agent, 'dqn_pendulum.pth')

    # For Pendulum visualization, make sure you have the trained model 'dqn_pendulum.pth'
    # env, state_size, action_size, discrete = create_env('Pendulum-v1')
    # agent = initialize_agent(state_size, action_size, discrete)
    # visualize_agent(env, agent, 'dqn_pendulum.pth')

if __name__ == "__main__":
    main()
