import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn   
# Define the Q-network architecture
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

# Evaluate the trained model
def evaluate_model(env_name, model_path, seed=0, n_episodes=1000):
    env = gym.make(env_name)
    env.reset(seed=seed)

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    qnetwork = QNetwork(state_size, action_size)
    qnetwork.load_state_dict(torch.load(model_path))
    qnetwork.eval()

    scores = []
    for _ in range(n_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        total_reward = 0
        done = False
        while not done:
            state = torch.from_numpy(state).float().unsqueeze(0)
            with torch.no_grad():
                action_values = qnetwork(state)
            action = np.argmax(action_values.cpu().data.numpy())
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            done = terminated or truncated
            total_reward += reward
            state = next_state
        scores.append(total_reward)

    env.close()
    return np.mean(scores), np.std(scores)

# Evaluate the models
env_name = 'CartPole-v1'
model_paths = [
    'models/dqn_cartpole_lr_0.0001_gamma_0.99_tau_0.01_seed_0.pth',
    'models/dqn_cartpole_lr_0.0001_gamma_0.99_tau_0.01_seed_1.pth',
    'models/dqn_cartpole_lr_0.0001_gamma_0.99_tau_0.01_seed_2.pth',
    'models/dqn_cartpole_lr_0.0005_gamma_0.99_tau_0.01_seed_0.pth',
    'models/dqn_cartpole_lr_0.0005_gamma_0.99_tau_0.01_seed_1.pth',
    'models/dqn_cartpole_lr_0.0005_gamma_0.99_tau_0.01_seed_2.pth'
]

for model_path in model_paths:
    mean_score, std_score = evaluate_model(env_name, model_path)
    print(f"Model: {model_path} - Mean Score: {mean_score}, Std Dev: {std_score}")


# Collect results
results = {}
for model_path in model_paths:
    mean_score, std_score = evaluate_model(env_name, model_path)
    results[model_path] = (mean_score, std_score)
    print(f"Model: {model_path} - Mean Score: {mean_score}, Std Dev: {std_score}")

# Plot the results
fig, ax = plt.subplots(figsize=(12, 8))

# Extracting values for plotting
model_labels = list(results.keys())
mean_scores = [results[model][0] for model in model_labels]
std_scores = [results[model][1] for model in model_labels]

# Plotting
ax.barh(model_labels, mean_scores, xerr=std_scores, align='center', alpha=0.7, ecolor='black', capsize=10)
ax.set_xlabel('Mean Score')
ax.set_title('Model Evaluation on CartPole-v1')
ax.invert_yaxis()  # labels read top-to-bottom

# Display the plot
plt.tight_layout()
plt.show()

