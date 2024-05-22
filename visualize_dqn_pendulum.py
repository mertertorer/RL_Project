import gym
import torch
import torch.nn as nn
import numpy as np

# Define the Q-network (same as the training script)
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

# DQN Agent (same as the training script)
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.qnetwork_local = QNetwork(state_size, action_size)

    def act(self, state, eps=0.0):
        state = torch.from_numpy(state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()
        action = np.argmax(action_values.cpu().data.numpy())
        return np.array([action])  # Ensure the action is an array for continuous environments

# Function to visualize the trained agent
def visualize_agent(env, agent, model_path, n_episodes=5):
    agent.qnetwork_local.load_state_dict(torch.load(model_path))
    agent.qnetwork_local.eval()  # Set the network to evaluation mode
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # Extract the actual state from the tuple
        total_reward = 0
        done = False
        while not done:
            env.render()
            action = agent.act(state, eps=0.0)  # Use a greedy policy (no exploration)
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]  # Extract the actual next state from the tuple
            done = terminated or truncated
            state = next_state
            total_reward += reward
        print(f"Episode {i_episode}\tTotal Reward: {total_reward}")
    
    env.close()

# Create the environment
env = gym.make('Pendulum-v1',  render_mode='human')

# Initialize the agent
state_size = env.observation_space.shape[0]
action_size = env.action_space.shape[0]
agent = DQNAgent(state_size, action_size)

# Visualize the agent
visualize_agent(env, agent, 'dqn_pendulum.pth')
