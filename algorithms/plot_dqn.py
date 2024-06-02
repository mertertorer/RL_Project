import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_scores(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


loaded_results = {}
for file_name in os.listdir('scores'):
    if file_name.startswith('scores_lr') and file_name.endswith('_pendulum.json'):
        key = file_name.replace('.json', '')
        loaded_results[key] = load_scores(os.path.join('scores', file_name))

def calculate_mean_std(results):
    mean_scores = {}
    std_scores = {}
    grouped_results = {}

    for key, scores in results.items():
        base_key = '_'.join(key.split('_')[:-2])
        if base_key not in grouped_results:
            grouped_results[base_key] = []
        grouped_results[base_key].append(scores)

    for base_key, scores_list in grouped_results.items():
        scores_array = np.array(scores_list)
        mean_scores[base_key] = np.mean(scores_array, axis=0)
        std_scores[base_key] = np.std(scores_array, axis=0)

    return mean_scores, std_scores

mean_scores, std_scores = calculate_mean_std(loaded_results)

best_key = max(mean_scores, key=lambda k: np.mean(mean_scores[k][-100:]))

plt.figure(figsize=(12, 8))
for key in mean_scores.keys():
    mean = mean_scores[key]
    std = std_scores[key]
    plt.plot(mean, label=key, alpha=0.6)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1)

best_mean = mean_scores[best_key]
best_std = std_scores[best_key]
plt.plot(best_mean, label=f'Best: {best_key}', color='black', linewidth=2)
plt.fill_between(range(len(best_mean)), best_mean - best_std, best_mean + best_std, color='black', alpha=0.3)

plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN Training Performance on Pendulum-v1 with Different Hyperparameters (Averaged over 3 seeds)')
plt.legend()
plt.show()

print(f'Best-performing hyperparameter combination: {best_key}')


##############
import matplotlib.pyplot as plt
import numpy as np
import json
import os

# load scores from JSON files
def load_scores(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

# load all scores into a dictionary
loaded_results = {}
for file_name in os.listdir('scores'):
    if file_name.startswith('scores_lr') and file_name.endswith('_pendulum.json'):
        key = file_name.replace('.json', '')
        loaded_results[key] = load_scores(os.path.join('scores', file_name))

# calculate the average and std of scores
def calculate_mean_std(results):
    mean_scores = {}
    std_scores = {}
    grouped_results = {}

    for key, scores in results.items():
        base_key = '_'.join(key.split('_')[:-2])  # Remove the seed part to group the results
        if base_key not in grouped_results:
            grouped_results[base_key] = []
        grouped_results[base_key].append(scores)

    for base_key, scores_list in grouped_results.items():
        scores_array = np.array(scores_list)
        mean_scores[base_key] = np.mean(scores_array, axis=0)
        std_scores[base_key] = np.std(scores_array, axis=0)

    return mean_scores, std_scores

#  mean and std of scores
mean_scores, std_scores = calculate_mean_std(loaded_results)

# best-performing hyperparam combination
best_key = max(mean_scores, key=lambda k: np.mean(mean_scores[k][-100:]))

# Plot average scores with standard deviation as shaded area
plt.figure(figsize=(12, 8))

# 3 best-performing configurations for clarity
top_n = 3
sorted_keys = sorted(mean_scores.keys(), key=lambda k: np.mean(mean_scores[k][-100:]), reverse=True)[:top_n]

for key in sorted_keys:
    mean = mean_scores[key]
    std = std_scores[key]
    plt.plot(mean, label=key, alpha=0.6)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.1)

# best-performing hyperparam combination
best_mean = mean_scores[best_key]
best_std = std_scores[best_key]
plt.plot(best_mean, label=f'Best: {best_key}', color='black', linewidth=2)
plt.fill_between(range(len(best_mean)), best_mean - best_std, best_mean + best_std, color='black', alpha=0.3)

plt.xlabel('Episode')
plt.ylabel('Score')
plt.title('DQN Training Performance on Pendulum-v1 with Different Hyperparameters (Averaged over 3 seeds)')
plt.legend()
plt.show()

print(f'Best-performing hyperparameter combination: {best_key}')


