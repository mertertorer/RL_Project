from algorithms.ppo import PPOAgent
import gym
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import datetime
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
    env = gym.make(env_name, render_mode="human")
    agent = PPOAgent(env, env_name=env_name,episodes=episodes, max_timesteps=max_timesteps)
    
    # Load the trained model parameters
    agent.load_model(path)
    
    # Visualize the agent's performance
    agent.visualize(iter = 20)

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
    plt.fill_between(episodes, avg_performance - std_performance, avg_performance + std_performance, alpha=0.2)
    plt.xlabel("Episode")
    plt.ylabel("Performance Metric")
    if reward_type == "rewards":
        path = f"plots/{seed_name}_E_{num_episodes}_TS_{max_timesteps}_{lr}_performance_metrics.png"
        plt.title("Algorithm Performance Averaged Over Seeds")
    else:
        path = f"plots/{seed_name}_E_{num_episodes}_TS_{max_timesteps}_{lr}_performance_metrics_last.png"
        plt.title("Algorithm Performance on Last Step Averaged Over Seeds") 
    plt.legend()
    plt.savefig(path)
    plt.close()



if __name__ == "__main__":
    seeds = [4,11,9]
   #train_multiple_seeds(seed_name,env_name,seeds,num_episodes,max_timesteps,lr):
    
    # train_multiple_seeds("pendulum1","Pendulum-v1",seeds,8000,400,1e-4)
    # train_multiple_seeds("pendulum2","Pendulum-v1",seeds,8000,200,5e-4)
    train_multiple_seeds("pendulum8","Pendulum-v1",seeds,4000,400,5e-5)
    # train_multiple_seeds("pendulum4","Pendulum-v1",[4,11,9,23],2000,500,1e-3)
    # train_multiple_seeds("pendulum5","Pendulum-v1",[4,11,9,23],2000,500,5e-4)
    # train_multiple_seeds("pendulum6","Pendulum-v1",[4,11,9,23],2000,500,1e-4)
    

    # test_train_agent('Pendulum-v1', episodes=4000, max_timesteps=400, lr = 5e-4)
    # test_train_agent('Pendulum-v1', episodes=4000, max_timesteps=400, lr = 1e-4)

    # test_train_agent('Pendulum-v1', episodes=10000, max_timesteps=400)
    # test_train_agent('Pendulum-v1', episodes=10000, max_timesteps=600)
    # test_train_agent('Pendulum-v1', episodes=20000, max_timesteps=200)
    # test_train_agent('Pendulum-v1', episodes=20000, max_timesteps=400)
    # test_train_agent('Pendulum-v1', episodes=20000, max_timesteps=400)
    
    #test_train_agent('CartPole-v1', episodes=1000, max_timesteps=500)
    #test_train_agent('CartPole-v1')
    # visualizing the trained agent for 10 times  
    #test_visualize_agent('models/ppo_model_Pendulum-v1_E_20000_TS_400_2024-05-25_07-13-46.pth',episodes = 20000 ,max_timesteps= 400, env_name='Pendulum-v1')
    # test_visualize_agent('ppo_model_cartpole.pth', max_timesteps= 10, env_name='CartPole-v1')
