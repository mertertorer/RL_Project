# ppo.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import datetime
import matplotlib.pyplot as plt
import os
import time
from torch.cuda.amp import GradScaler, autocast

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

class PPOAgent:
    def __init__(self, env, gamma=0.99, lr=3e-4, eps_clip=0.2, K_epochs=4, lambd=0.95):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambd = lambd
        self.memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = GradScaler()

    def select_action(self, state):
        if isinstance(state, tuple):
            state = state[0]
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, value = self.model(state)
        action = torch.tanh(policy).detach().cpu().numpy().flatten()
        log_prob = -0.5 * ((torch.FloatTensor(action).to(self.device) - policy) ** 2).sum(dim=1)
        return action, value.item(), log_prob.item()

    def save_model(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = f"ppo_model_{current_time}.pth"
        try:
            print(f"Saving model to: {os.path.abspath(path)}")  # Print the absolute path
            torch.save(self.model.state_dict(), path)
            print("Model saved to", path)
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))
        print("Model loaded from", path)

    def store_transition(self, transition):
        self.memory.append(transition)

    def clear_memory(self):
        self.memory = []

    def collect_trajectories(self, max_timesteps):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        for t in range(max_timesteps):
            action, value, log_prob = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            mask = 1 - done
            self.store_transition((state, action, reward, value, mask, log_prob))
            state = next_state
            if done:
                state = self.env.reset()
                if isinstance(state, tuple):
                    state = state[0]

    def compute_gae(self, rewards, values, masks, next_value):
        values = values + (next_value,)  # Convert list to tuple before concatenating
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + self.gamma * self.lambd * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    def optimize(self):
        states, actions, rewards, values, masks, old_log_probs = zip(*self.memory)
        next_value = self.model(torch.FloatTensor(states[-1]).unsqueeze(0).to(self.device))[1].item()
        returns = self.compute_gae(rewards, values, masks, next_value)

        returns = torch.FloatTensor(returns).to(self.device)
        advantages = returns - torch.FloatTensor(values).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)

        for _ in range(self.K_epochs):
            for state, action, return_, advantage, old_log_prob in zip(states, actions, returns, advantages, old_log_probs):
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action = torch.FloatTensor(action).unsqueeze(0).to(self.device)
                return_ = torch.FloatTensor([return_]).to(self.device)
                advantage = torch.FloatTensor([advantage]).to(self.device)
                old_log_prob = torch.FloatTensor([old_log_prob]).to(self.device)

                policy, value = self.model(state)
                log_prob = -0.5 * ((action - policy) ** 2).sum(dim=1, keepdim=True)
                ratio = torch.exp(log_prob - old_log_prob)
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(value.squeeze(), return_.squeeze())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        self.memory = []

    def train(self, episodes, max_timesteps):
        rewards_per_episode = []
        start_time = time.time()  # Capture start time of training
        for episode in range(episodes):
            episode_rewards = 0
            self.collect_trajectories(max_timesteps)
            self.optimize()
            for transition in self.memory:
                _, _, reward, _, _, _ = transition
                episode_rewards += reward
            rewards_per_episode.append(episode_rewards)
            self.clear_memory()
            print(f"Episode {episode + 1}/{episodes} completed, Total Reward: {episode_rewards}")
        
        # Plotting the rewards
        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Rewards')
        plt.show()
        # save the model
        self.save_model()
        end_time = time.time()
        print(f"Training completed in {end_time - start_time} seconds")


    def visualize(self, max_timesteps):
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        for t in range(max_timesteps):
            self.env.render()
            action, _, _ = self.select_action(state)
            next_state, _, done, _, _ = self.env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            state = next_state
            if done:
                break
        self.env.close()