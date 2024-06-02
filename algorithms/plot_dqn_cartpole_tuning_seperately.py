import json
import matplotlib.pyplot as plt
import os

# Load the results
results = {}
for file_name in os.listdir('.'):
    if file_name.startswith('scores_lr') and file_name.endswith('.json'):
        key = file_name.replace('.json', '')
        with open(file_name, 'r') as f:
            results[key] = json.load(f)

# Plotting the results separately
num_plots = len(results)
plt.figure(figsize=(20, num_plots * 4))  # Adjust the figure size accordingly

for i, (key, scores) in enumerate(results.items()):
    plt.subplot(num_plots, 1, i + 1)
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title(f'DQN Training Performance on CartPole-v1: {key}')

plt.tight_layout()
plt.show()


import json
import matplotlib.pyplot as plt
import os
import numpy as np

# Load the results
results = {}
for file_name in os.listdir('.'):
    if file_name.startswith('scores_lr') and file_name.endswith('.json'):
        key = file_name.replace('.json', '')
        with open(file_name, 'r') as f:
            results[key] = json.load(f)

# Calculate the average score for the last 100 episodes for each configuration
avg_scores = {key: np.mean(scores[-100:]) for key, scores in results.items()}

# Sort the configurations by average score
sorted_avg_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)

# Select the top 2 configurations
top_2_configs = sorted_avg_scores[:2]

# Plotting the results for the top 3 configurations
plt.figure(figsize=(12, 8))
for key, _ in top_2_configs:
    plt.plot(results[key], label=key)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Top 2 DQN Training Performance on CartPole-v1 with Different Hyperparameters')
plt.legend()
plt.show()



import json
import matplotlib.pyplot as plt
import numpy as np

# Load scores from JSON files
def load_scores(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Paths to the score files
score_files = [
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_0.json",
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_1.json",
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_2.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_0.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_1.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_2.json"
]

# Load the scores
scores = {filepath: load_scores(filepath) for filepath in score_files}

# Calculate average scores for each configuration
def calculate_average(scores, keys):
    scores_array = np.array([scores[key] for key in keys])
    return np.mean(scores_array, axis=0)

# Keys for each configuration
keys_config1 = [
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_0.json",
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_1.json",
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_2.json"
]

keys_config2 = [
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_0.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_1.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_2.json"
]

# Calculate the average scores
average_scores_config1 = calculate_average(scores, keys_config1)
average_scores_config2 = calculate_average(scores, keys_config2)

# Plotting the results
plt.figure(figsize=(12, 8))
plt.plot(average_scores_config1, label="lr=0.0005, gamma=0.99, tau=0.01")
plt.plot(average_scores_config2, label="lr=0.0001, gamma=0.99, tau=0.01")
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Average DQN Training Performance on CartPole-v1 with Different Hyperparameters')
plt.legend()
plt.show()

##########################
import json
import json
import numpy as np
import matplotlib.pyplot as plt

# Load scores from JSON files
def load_scores(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Paths to the score files for the two best configurations
score_files_config1 = [
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_0.json",
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_1.json",
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_2.json"
]

score_files_config2 = [
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_0.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_1.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_2.json"
]

# Load the scores
scores_config1 = [load_scores(filepath) for filepath in score_files_config1]
scores_config2 = [load_scores(filepath) for filepath in score_files_config2]

# Calculate the average and standard deviation of the scores
mean_config1 = np.mean(scores_config1, axis=0)
std_config1 = np.std(scores_config1, axis=0)

mean_config2 = np.mean(scores_config2, axis=0)
std_config2 = np.std(scores_config2, axis=0)

# Overall mean and standard deviation
overall_mean_config1 = np.mean(mean_config1)
overall_std_config1 = np.mean(std_config1)

overall_mean_config2 = np.mean(mean_config2)
overall_std_config2 = np.mean(std_config2)

# Print mean and standard deviation for comparison
print(f"Configuration 1 (lr=0.0005, gamma=0.99, tau=0.01): Mean Score = {overall_mean_config1}, Std Dev = {overall_std_config1}")
print(f"Configuration 2 (lr=0.0001, gamma=0.99, tau=0.01): Mean Score = {overall_mean_config2}, Std Dev = {overall_std_config2}")

# Plotting the average scores with standard deviation as shaded area
plt.figure(figsize=(12, 8))
line1, = plt.plot(mean_config1, label="lr=0.0005, gamma=0.99, tau=0.01")
plt.fill_between(range(len(mean_config1)), mean_config1 - std_config1, mean_config1 + std_config1, alpha=0.2)
line2, = plt.plot(mean_config2, label="lr=0.0001, gamma=0.99, tau=0.01")
plt.fill_between(range(len(mean_config2)), mean_config2 - std_config2, mean_config2 + std_config2, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Average DQN Training Performance on CartPole-v1 with Different Hyperparameters')
plt.legend()

# Adding mean and std info to the plot near the end of the lines
plt.gca().annotate(f'Mean: {overall_mean_config1:.2f}\nStd Dev: {overall_std_config1:.2f}', 
                   xy=(len(mean_config1)-1, mean_config1[-1]), 
                   xycoords='data', 
                   fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor=line1.get_color(), facecolor="white"),
                   verticalalignment='bottom')

plt.gca().annotate(f'Mean: {overall_mean_config2:.2f}\nStd Dev: {overall_std_config2:.2f}', 
                   xy=(len(mean_config2)-1, mean_config2[-1]), 
                   xycoords='data', 
                   fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor=line2.get_color(), facecolor="white"),
                   verticalalignment='top')

plt.show()


###############################33
import json
import numpy as np
import matplotlib.pyplot as plt

# Load scores from JSON files
def load_scores(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Paths to the score files for the two best configurations
score_files_config1 = [
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_0.json",
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_1.json",
    "scores_lr_0.0005_gamma_0.99_tau_0.01_seed_2.json"
]

score_files_config2 = [
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_0.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_1.json",
    "scores_lr_0.0001_gamma_0.99_tau_0.01_seed_2.json"
]

# Load the scores
scores_config1 = [load_scores(filepath) for filepath in score_files_config1]
scores_config2 = [load_scores(filepath) for filepath in score_files_config2]

# Calculate the average and standard deviation of the scores
mean_config1 = np.mean(scores_config1, axis=0)
std_config1 = np.std(scores_config1, axis=0)

mean_config2 = np.mean(scores_config2, axis=0)
std_config2 = np.std(scores_config2, axis=0)

# Overall mean and standard deviation
overall_mean_config1 = np.mean(mean_config1)
overall_std_config1 = np.mean(std_config1)

overall_mean_config2 = np.mean(mean_config2)
overall_std_config2 = np.mean(std_config2)

# Print mean and standard deviation for comparison
print(f"Configuration 1 (lr=0.0005, gamma=0.99, tau=0.01): Mean Score = {overall_mean_config1}, Std Dev = {overall_std_config1}")
print(f"Configuration 2 (lr=0.0001, gamma=0.99, tau=0.01): Mean Score = {overall_mean_config2}, Std Dev = {overall_std_config2}")

# Plotting the average scores with standard deviation as shaded area
plt.figure(figsize=(12, 8))
line1, = plt.plot(mean_config1, label="lr=0.0005, gamma=0.99, tau=0.01")
plt.fill_between(range(len(mean_config1)), mean_config1 - std_config1, mean_config1 + std_config1, alpha=0.2)
line2, = plt.plot(mean_config2, label="lr=0.0001, gamma=0.99, tau=0.01")
plt.fill_between(range(len(mean_config2)), mean_config2 - std_config2, mean_config2 + std_config2, alpha=0.2)
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('Average DQN Training Performance on CartPole-v1 with Different Hyperparameters\n(Averaged over 3 seeds)')
plt.legend()

# Adding mean and std info to the plot near the end of the lines
plt.gca().annotate(f'Mean: {overall_mean_config1:.2f}\nStd Dev: {overall_std_config1:.2f}', 
                   xy=(len(mean_config1)-1, mean_config1[-1]), 
                   xycoords='data', 
                   fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor=line1.get_color(), facecolor="white"),
                   verticalalignment='bottom')

plt.gca().annotate(f'Mean: {overall_mean_config2:.2f}\nStd Dev: {overall_std_config2:.2f}', 
                   xy=(len(mean_config2)-1, mean_config2[-1]), 
                   xycoords='data', 
                   fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor=line2.get_color(), facecolor="white"),
                   verticalalignment='top')

plt.show()


######
import json
import matplotlib.pyplot as plt
import numpy as np

# Load scores from JSON files
def load_scores(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# Paths to the score files for different seeds
score_files = [
    "scores_pendulum_seed_0.json",
    "scores_pendulum_seed_1.json",
    "scores_pendulum_seed_2.json"
]

# Load the scores
scores = [load_scores(filepath) for filepath in score_files]

# Calculate the average and standard deviation of the scores
mean_scores = np.mean(scores, axis=0)
std_scores = np.std(scores, axis=0)

# Overall mean and standard deviation
overall_mean = np.mean(mean_scores)
overall_std = np.mean(std_scores)

# Print mean and standard deviation for reference
print(f"Mean Score: {overall_mean}, Std Dev: {overall_std}")

# Plotting the results
plt.figure(figsize=(12, 8))
for i, score in enumerate(scores):
    plt.plot(score, label=f'seed_{i}')
plt.plot(mean_scores, label="Average Score", color='black', linestyle='--')
plt.fill_between(range(len(mean_scores)), mean_scores - std_scores, mean_scores + std_scores, alpha=0.2, label="Std Dev", color='gray')
plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN Training Performance on Pendulum-v1\nHyperparameters: lr=0.0001, gamma=0.99, tau=0.01 (Averaged over 3 seeds)')
plt.legend()

# Adding mean and std info to the plot
plt.gca().annotate(f'Mean: {overall_mean:.2f}\nStd Dev: {overall_std:.2f}', 
                   xy=(len(mean_scores)-1, mean_scores[-1]), 
                   xycoords='data', 
                   fontsize=12, 
                   bbox=dict(boxstyle="round,pad=0.3", edgecolor='black', facecolor="white"),
                   verticalalignment='top')

plt.show()

