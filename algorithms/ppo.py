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

        # TODO: You might have different actor for continuous and discrete action spaces
        self.actor = nn.Sequential( 
            nn.Linear(input_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        policy = self.actor(x)
        value = self.critic(x)
        return policy, value

class PPOAgent:
    def __init__(self, env, gamma=0.99, lr=5e-4, eps_clip=0.2, K_epochs=10, lambd=0.95, device="cpu"):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.lambd = lambd
        self.memory = []
        self.device = torch.device("cuda" if torch.cuda.is_available() and device=="cuda" else "cpu")
        self.model = ActorCritic(env.observation_space.shape[0], env.action_space.shape[0]).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scaler = GradScaler()

    def select_action(self, state):
        min_action = self.env.action_space.low
        max_action = self.env.action_space.high
        if isinstance(state, tuple):
            state = state[0]
        if not isinstance(state, np.ndarray):
            state = np.array(state)
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        policy, value = self.model(state)
        action = (torch.tanh(policy).detach().cpu().numpy().flatten()+1)/2 * (max_action - min_action) + min_action
        log_prob = -0.5 * ((torch.FloatTensor(action).to(self.device) - policy) ** 2).sum(dim=1)
        return action, value.item(), log_prob.item()

    def save_model(self):
        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        path = "ppo_model.pth" #f"ppo_model_{current_time}.pth"
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
        episode_rewards = []
        for t in range(max_timesteps):
            action, value, log_prob = self.select_action(state)
            next_state, reward, done, _ , _= self.env.step(action)
            # reward = reward/16.2736044 # Normalize reward
            #give reward for positive y and penalize for negative y 
           
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            mask = 1 - done
            self.store_transition((state, action, reward, value, mask, log_prob))
            state = next_state
            episode_rewards.append(reward)
            if done:
                state = self.env.reset()
                if isinstance(state, tuple):
                    state = state[0]

        # Normalize rewards
        
        # mean_reward = np.mean(episode_rewards)
        # std_reward = np.std(episode_rewards)
        # for i in range(len(self.memory)):
        #     self.memory[i] = (self.memory[i][0], self.memory[i][1], (self.memory[i][2] - mean_reward) / (std_reward + 1e-8),
        #                       self.memory[i][3], self.memory[i][4], self.memory[i][5])

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

        batch_size = 32  # Set your batch size here
        num_batches = (len(states) + batch_size - 1) // batch_size

        
        for _ in range(self.K_epochs): 
            for i in range(num_batches):
               
                start = i * batch_size
                end = start + batch_size
                if end > len(states):
                    end = len(states)

                state = torch.tensor(states[start:end], dtype=torch.float32).to(self.device)
                action = torch.tensor(actions[start:end], dtype=torch.float32).to(self.device)
                return_ = torch.tensor(returns[start:end], dtype=torch.float32).to(self.device)
                advantage = torch.tensor(advantages[start:end], dtype=torch.float32).to(self.device)
                old_log_prob = torch.tensor(old_log_probs[start:end], dtype=torch.float32).to(self.device)

                policy, value = self.model(state)
                log_prob = -0.5 * ((action - policy) ** 2).sum(dim=1, keepdim=True)
                
                # Check for NaNs in log_prob
                if torch.isnan(log_prob).any():
                    print("NaN detected in log_prob")
                    print("log_prob:", log_prob)
                    continue

                ratio = torch.exp(log_prob - old_log_prob)
                
                # Check for NaNs in ratio
                if torch.isnan(ratio).any():
                    print("NaN detected in ratio")
                    print("ratio:", ratio)
                    continue

                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * advantage
                
                # Check for NaNs in surr1 and surr2
                if torch.isnan(surr1).any() or torch.isnan(surr2).any():
                    print("NaN detected in surr1 or surr2")
                    print("surr1:", surr1)
                    print("surr2:", surr2)
                    continue

                loss = -torch.min(surr1, surr2).mean() + 0.5 * nn.MSELoss()(value.squeeze(), return_.squeeze())

                # Check for NaNs in loss
                if torch.isnan(loss).any():
                    print("NaN detected in loss")
                    print("loss:", loss)
                    continue

                self.optimizer.zero_grad()
                loss.backward()

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)

                self.optimizer.step()

        self.memory = []

    def train(self, episodes, max_timesteps):
        rewards_per_episode = []
        start_time = time.time()  # Capture start time of training
        for episode in range(episodes):
            episode_rewards = 0
            self.collect_trajectories(max_timesteps)
            for transition in self.memory:
                _, _, reward, _, _, _ = transition
                episode_rewards += reward
            rewards_per_episode.append(episode_rewards)
            self.optimize()
            self.clear_memory()
            if (episode + 1) % 100 == 0 or episode == 0:
                print(f"Episode {episode + 1}/{episodes} completed, Total Reward: {episode_rewards}")
        
        # Plotting the rewards
        
        end_time = time.time()
        print(f"Training completed in {end_time - start_time} seconds")
        # save the model
        self.save_model()

        plt.plot(rewards_per_episode)
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title('Training Rewards')
        plt.show()

    def visualize(self, max_timesteps=200):      
        state = self.env.reset()
        if isinstance(state, tuple):
            state = state[0]
        for t in range(max_timesteps):
            self.env.render()
            action, _, _ = self.select_action(state)
            next_state, reward, done, _, _ = self.env.step(action)
            if state[0] <= 0.2:
                reward = reward -0.5
            else:
                if np.abs(state[2]) > 3:
                    reward = reward - 0.5
            print("Action:", action, "State:", state, "Reward:", reward)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            state = next_state
            if done:
                break
        self.env.close()
