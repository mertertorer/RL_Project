from algorithms.ppo import PPOAgent
import gym
import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the name of the GPU
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def test_train_agent(env_name, episodes=4000, max_timesteps=500, lr = 5e-4):
    env = gym.make(env_name)
    agent = PPOAgent(env, device="cpu", env_name=env_name,episodes=episodes, max_timesteps=max_timesteps, lr = lr)
    rewards_per_episode, last_rewards_per_episode = agent.train()
    print("Training completed")
    return rewards_per_episode, last_rewards_per_episode

def test_visualize_agent(path, episodes, max_timesteps, env_name='Pendulum-v1'):
    # comment out render_mode="human" to run vithout visualization
    env = gym.make(env_name, render_mode="human")
    agent = PPOAgent(env, env_name=env_name,episodes=episodes, max_timesteps=500)
    
    # Load the trained model parameters
    agent.load_model(path)
    
    # Visualize the agent's performance
    agent.visualize(iter = 200)

def train_multiple_seeds(seed_name,env_name,seeds,num_episodes,max_timesteps,lr):
    all_performances =[]
    all_performances_last = [] 
    for seed in seeds:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        rewards, last_rewards = test_train_agent(env_name, episodes=num_episodes, max_timesteps=max_timesteps, lr = lr)
        all_performances.append(rewards)
        all_performances_last.append(last_rewards)
        #save performance metrics and last performance metrics 
        path = f'perf_metrics/{seed_name}_S_{seed}_E_{num_episodes}_TS_{max_timesteps}_{lr}_rewards.npy'
        path_last = f'perf_metrics/{seed_name}_S_{seed}_E_{num_episodes}_TS_{max_timesteps}_{lr}_rewards_last.npy'
        np.save(path, rewards)
        np.save(path_last, last_rewards)

    plot_performances("rewards",all_performances, seed_name, num_episodes, max_timesteps, lr)
    plot_performances("rewards_last",all_performances_last, seed_name, num_episodes, max_timesteps, lr)

def plot_performances(reward_type ,all_performances, seed_name, num_episodes, max_timesteps, lr):
    avg_performance = np.mean(all_performances, axis=0).flatten()
    std_performance = np.std(all_performances, axis=0).flatten()

    # Correctly define the episodes array
    episodes = np.arange(avg_performance.shape[0])

    plt.figure()
    plt.plot(episodes, avg_performance, label="Average Performance")
    plt.axhline(y=0, color='r', linestyle='--')
    plt.fill_between(episodes, avg_performance - std_performance, avg_performance + std_performance, alpha=0.3)
    plt.xlabel("Episode")
    plt.ylabel("Performance Metric")
    if reward_type == "rewards":
        path = f"plots/{seed_name}_E_{num_episodes}_TS_{max_timesteps}_{lr}_performance_metrics.png"
        plt.title("Algorithm Performance Averaged Over Seeds")
    else:
        path = f"plots/{seed_name}_E_{num_episodes}_TS_{max_timesteps}_{lr}_performance_metrics_last.png"
        plt.title("Algorithm Performance on Last Step Averaged Over Seeds") 
    plt.legend()
    plt.show()
    plt.savefig(path)
    plt.close()


def plot_performances_multiple(data_list):
    plt.figure()

    for data in data_list:
        reward_type, all_performances, seed_name, num_episodes, max_timesteps, lr = data
        #all_performances = all_performances / max_timesteps
        # Calculate the mean and standard deviation for each 100-step block
        block_size = 100
        avg_performance = np.mean(all_performances, axis=0).flatten()
        std_performance = np.std(all_performances, axis=0).flatten()

        num_blocks = len(avg_performance) // block_size
        avg_performance_blocks = [np.mean(avg_performance[i*block_size:(i+1)*block_size]) for i in range(num_blocks)]
        std_performance_blocks = [np.mean(std_performance[i*block_size:(i+1)*block_size]) for i in range(num_blocks)]

        # Define the episodes array for the blocks
        episodes_blocks = np.arange(num_blocks) * block_size

        if reward_type == "rewards":
            label = f"{seed_name}"
        else:
            label = f"Last Step Performance {seed_name}"
        
        plt.plot(episodes_blocks, avg_performance_blocks, label=label)
        plt.fill_between(episodes_blocks, np.array(avg_performance_blocks) - np.array(std_performance_blocks), np.array(avg_performance_blocks) + np.array(std_performance_blocks), alpha=0.2)
    
    
    plt.axhline(y=500, color='r', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Average Algorithm Performance - 3 Different Seeds (averaged over 100 steps)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    seeds = [4,11,9]
    
    #train_multiple_seeds("cartpole1","CartPole-v1",seeds,1000,500,1e-4)

    #test_visualize_agent('models/ppo_model_Pendulum-v1_E_10000_TS_320_2024-05-30_06-54-50.pth',episodes = 10000 ,max_timesteps= 320, env_name='Pendulum-v1')

    # Loading performances for each seed
    # perf1 = np.load("perf_metrics/2024-05-27_04-55-15_S_2_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_04-30-15_S_0_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_04-42-46_S_1_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # all_performances_1 = np.array([perf1, perf2, perf3])
    # data1 = ("rewards", all_performances_1, "Ep: 2711, TS: 311, lr: 2.5e-4", 2711, 311, 0.0002515289269330568)

    # perf4 = np.load("perf_metrics/2024-05-27_05-07-52_S_0_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # perf5 = np.load("perf_metrics/2024-05-27_05-20-32_S_1_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # perf6 = np.load("perf_metrics/2024-05-27_05-33-10_S_2_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # all_performances_2 = np.array([perf4, perf5, perf6])
    # data2 = ("rewards", all_performances_2, "Ep: 2469, TS: 366, lr: 2e-5", 2469, 366, 2.0795089871175035e-05)

    # all_performances = [data1, data2]
    # plot_performances_multiple(all_performances)
