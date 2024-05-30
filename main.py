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
    agent = PPOAgent(env, env_name=env_name,episodes=episodes, max_timesteps=200)
    
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

# create plot performance metrics function with multiple all_performances and all_performances_last as input and plot them with different colors with the same plot
# input parameters are 5 different arrays of [reward_type ,all_performances, seed_name, num_episodes, max_timesteps, lr]

def plot_performances_multiple(data_list):
    plt.figure()

    for data in data_list:
        reward_type, all_performances, seed_name, num_episodes, max_timesteps, lr = data
        all_performances = all_performances / max_timesteps
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
    
    
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Average Algorithm Performance - 3 Different Seeds (averaged over 100 steps)")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    seeds = [4,11,9]
   #train_multiple_seeds(seed_name,env_name,seeds,num_episodes,max_timesteps,lr):
    
    train_multiple_seeds("cartpole1","CartPole-v1",seeds,1000,500,1e-4)
    #train_multiple_seeds("cartpole2","CartPole-v1",seeds,1000,500,2.5e-4)
    #train_multiple_seeds("cartpole3","CartPole-v1",seeds,1000,500,5e-5)
    # train_multiple_seeds("pendulum12","Pendulum-v1",seeds,10000,320,2.5e-4)
    # train_multiple_seeds("pendulum13","Pendulum-v1",seeds,10000,192,5e-4)
    # train_multiple_seeds("pendulum14","Pendulum-v1",seeds,10000,192,2e-4)
    # train_multiple_seeds("pendulum15","Pendulum-v1",seeds,10000,256,2e-4) #4.6h
    # train_multiple_seeds("pendulum16","Pendulum-v1",seeds,10000,384,2e-4) #6.2h
    # train_multiple_seeds("pendulum17","Pendulum-v1",seeds,20000,256,1e-4)
    # train_multiple_seeds("pendulum18","Pendulum-v1",seeds,20000,384,1e-4)
    # train_multiple_seeds("pendulum2","Pendulum-v1",seeds,8000,200,5e-4)
    # train_multiple_seeds("pendulum8","Pendulum-v1",seeds,4000,400,5e-5)
    # train_multiple_seeds("pendulum4","Pendulum-v1",[4,11,9,23],2000,500,1e-3)
    # train_multiple_seeds("pendulum5","Pendulum-v1",[4,11,9,23],2000,500,5e-4)
    # train_multiple_seeds("pendulum6","Pendulum-v1",[4,11,9,23],2000,500,1e-4)
    
    #test_visualize_agent('models/ppo_model_Pendulum-v1_E_10000_TS_320_2024-05-30_06-54-50.pth',episodes = 10000 ,max_timesteps= 320, env_name='Pendulum-v1')
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

    # # Loading performances for each seed
    # perf1 = np.load("perf_metrics/2024-05-27_04-55-15_S_2_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_04-30-15_S_0_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_04-42-46_S_1_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # all_performances_1 = np.array([perf1, perf2, perf3])

    # perf4 = np.load("perf_metrics/2024-05-27_05-07-52_S_0_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # perf5 = np.load("perf_metrics/2024-05-27_05-20-32_S_1_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # perf6 = np.load("perf_metrics/2024-05-27_05-33-10_S_2_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # all_performances_2 = np.array([perf4, perf5, perf6])

    # perf7 = np.load("perf_metrics/2024-05-27_05-49-10_S_0_E_2872_TS_366_1.6280798813484343e-05_rewards.npy")
    # perf8 = np.load("perf_metrics/2024-05-27_06-05-11_S_1_E_2872_TS_366_1.6280798813484343e-05_rewards.npy")
    # perf9 = np.load("perf_metrics/2024-05-27_06-21-12_S_2_E_2872_TS_366_1.6280798813484343e-05_rewards.npy")
    # all_performances_3 = np.array([perf7, perf8, perf9])

    # perf10 = np.load("perf_metrics/2024-05-27_06-33-58_S_0_E_2288_TS_376_1.0544223184265175e-05_rewards.npy")
    # perf11 = np.load("perf_metrics/2024-05-27_06-46-43_S_1_E_2288_TS_376_1.0544223184265175e-05_rewards.npy")
    # perf12 = np.load("perf_metrics/2024-05-27_06-59-33_S_2_E_2288_TS_376_1.0544223184265175e-05_rewards.npy")
    # all_performances_4 = np.array([perf10, perf11, perf12])


    # # Load performances for each dataset
    # perf1_1 = np.load("perf_metrics/2024-05-27_04-55-15_S_2_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # perf1_2 = np.load("perf_metrics/2024-05-27_04-30-15_S_0_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # perf1_3 = np.load("perf_metrics/2024-05-27_04-42-46_S_1_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # all_performances1 = np.array([perf1_1, perf1_2, perf1_3])

    # perf2_1 = np.load("perf_metrics/old perf metrics/pendulum1_S_4_E_8000_TS_400_0.0001_rewards.npy")
    # perf2_2 = np.load("perf_metrics/old perf metrics/pendulum1_S_9_E_8000_TS_400_0.0001_rewards.npy")
    # perf2_3 = np.load("perf_metrics/old perf metrics/pendulum1_S_11_E_8000_TS_400_0.0001_rewards.npy")
    # all_performances2 = np.array([perf2_1, perf2_2, perf2_3])

    # perf3_1 = np.load("perf_metrics/old perf metrics/pendulum7_S_4_E_20000_TS_400_5e-05_rewards.npy")
    # perf3_2 = np.load("perf_metrics/old perf metrics/pendulum7_S_9_E_20000_TS_400_5e-05_rewards.npy")
    # perf3_3 = np.load("perf_metrics/old perf metrics/pendulum7_S_11_E_20000_TS_400_5e-05_rewards.npy")
    # all_performances3 = np.array([perf3_1, perf3_2, perf3_3])

    # perf4_1 = np.load("perf_metrics/old perf metrics/pendulum6_S_4_E_2000_TS_500_0.0001_rewards.npy")
    # perf4_2 = np.load("perf_metrics/old perf metrics/pendulum6_S_9_E_2000_TS_500_0.0001_rewards.npy")
    # perf4_3 = np.load("perf_metrics/old perf metrics/pendulum6_S_11_E_2000_TS_500_0.0001_rewards.npy")
    # all_performances4 = np.array([perf4_1, perf4_2, perf4_3])


    # data1 = ("rewards", all_performances_1, "Ep: 2711, TS: 311, lr: 2.5e-4", 2711, 311, 0.0002515289269330568)
    # data2 = ("rewards", all_performances_2, "Ep: 2469, TS: 366, lr: 2e-5", 2469, 366, 2.0795089871175035e-05)
    # data3 = ("rewards", all_performances_3, "Ep: 2872, TS: 366, lr: 1.6e-5", 2872, 366, 1.628079818348434e-05)
    # data4 = ("rewards", all_performances_4, "Ep: 2288, TS: 376, lr: 1e-5", 2288, 376, 1.0544223184265175e-05)

    # data5 = ("rewards", all_performances1, "Ep: 2711, TS: 311, lr: 2.5e-4", 2711, 311, 0.0002515289269330568)
    # data6 = ("rewards", all_performances2, "Ep: 8000, TS: 400, lr: 1e-4", 8000, 400, 0.0001)
    # data7 = ("rewards", all_performances3, "Ep: 20000, TS: 400, lr: 5e-5", 20000, 400, 5e-05)
    # data8 = ("rewards", all_performances4, "Ep: 2000, TS: 500, lr: 1e-4", 2000, 500, 0.0001)

    # all_performances = [data1, data2, data3, data4, data5, data6, data7, data8]


    # plot_performances_multiple(all_performances)

    # data1 = ("rewards", all_performances, "2024-05-27_04-55-15", 2711, 311, 0.0002515289269330568)
    # perf1_last = np.load("perf_metrics/2024-05-27_04-55-15_S_2_E_2711_TS_311_0.0002515289269330568_rewards_last.npy")
    # perf2_last = np.load("perf_metrics/2024-05-27_04-30-15_S_0_E_2711_TS_311_0.0002515289269330568_rewards_last.npy")
    # perf3_last = np.load("perf_metrics/2024-05-27_04-42-46_S_1_E_2711_TS_311_0.0002515289269330568_rewards_last.npy")
    # all_performances_last = np.array([perf1_last,perf2_last,perf3_last])
    
    # plot the performance metrics
    # plot_performances("rewards",all_performances, "2024-05-27_04-55-15", 2711, 311, 0.0002515289269330568)
    # plot_performances("rewards_last",all_performances_last, "2024-05-27_04-55-15", 2711, 311, 0.0002515289269330568)


    # # Load the performance data
    # perf1 = np.load("perf_metrics/2024-05-25_19-23-31_S_0_E_1965_TS_224_8.735679120657569e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-25_19-30-13_S_1_E_1965_TS_224_8.735679120657569e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-25_19-37-00_S_2_E_1965_TS_224_8.735679120657569e-05_rewards.npy")
    # all_performances_1 = np.array([perf1, perf2, perf3])
    # data1 = ("rewards", all_performances_1, "Ep: 1965, TS: 224, lr: 8.7e-5", 1965, 224, 8.735679120657569e-05)

    # perf1 = np.load("perf_metrics/2024-05-25_20-01-12_S_0_E_2927_TS_526_0.0003036083149394725_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-25_20-25-21_S_1_E_2927_TS_526_0.0003036083149394725_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-25_20-49-20_S_2_E_2927_TS_526_0.0003036083149394725_rewards.npy")
    # all_performances_2 = np.array([perf1, perf2, perf3])
    # data2 = ("rewards", all_performances_2, "Ep: 2927, TS: 526, lr: 3.0e-4", 2927, 526, 0.0003036083149394725)

    # perf1 = np.load("perf_metrics/2024-05-25_20-57-28_S_0_E_1208_TS_428_0.0001603361314327096_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-25_21-05-38_S_1_E_1208_TS_428_0.0001603361314327096_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-25_21-13-50_S_2_E_1208_TS_428_0.0001603361314327096_rewards.npy")
    # all_performances_3 = np.array([perf1, perf2, perf3])
    # data3 = ("rewards", all_performances_3, "Ep: 1208, TS: 428, lr: 1.6e-4", 1208, 428, 0.0001603361314327096)

    # perf1 = np.load("perf_metrics/2024-05-25_21-18-41_S_0_E_1316_TS_224_0.006768654184516114_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-25_21-23-33_S_1_E_1316_TS_224_0.006768654184516114_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-25_21-28-10_S_2_E_1316_TS_224_0.006768654184516114_rewards.npy")
    # all_performances_4 = np.array([perf1, perf2, perf3])
    # data4 = ("rewards", all_performances_4, "Ep: 1316, TS: 224, lr: 6.8e-3", 1316, 224, 0.006768654184516114)

    # perf1 = np.load("perf_metrics/2024-05-25_21-57-14_S_0_E_4459_TS_365_0.004884585808844383_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-25_22-26-50_S_1_E_4459_TS_365_0.004884585808844383_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-25_22-53-51_S_2_E_4459_TS_365_0.004884585808844383_rewards.npy")
    # all_performances_5 = np.array([perf1, perf2, perf3])
    # data5 = ("rewards", all_performances_5, "Ep: 4459, TS: 365, lr: 4.9e-3", 4459, 365, 0.004884585808844383)

    # perf1 = np.load("perf_metrics/2024-05-25_23-10-07_S_0_E_3014_TS_328_4.7866817258231506e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-25_23-26-20_S_1_E_3014_TS_328_4.7866817258231506e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-25_23-42-56_S_2_E_3014_TS_328_4.7866817258231506e-05_rewards.npy")
    # all_performances_6 = np.array([perf1, perf2, perf3])
    # data6 = ("rewards", all_performances_6, "Ep: 3014, TS: 328, lr: 4.8e-5", 3014, 328, 4.7866817258231506e-05)

    # perf1 = np.load("perf_metrics/2024-05-25_23-51-15_S_0_E_1037_TS_497_2.1851481565238554e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-25_23-59-42_S_1_E_1037_TS_497_2.1851481565238554e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-26_00-08-56_S_2_E_1037_TS_497_2.1851481565238554e-05_rewards.npy")
    # all_performances_7 = np.array([perf1, perf2, perf3])
    # data7 = ("rewards", all_performances_7, "Ep: 1037, TS: 497, lr: 2.2e-5", 1037, 497, 2.1851481565238554e-05)

    # perf1 = np.load("perf_metrics/2024-05-26_00-28-14_S_0_E_2563_TS_335_0.00562827575170271_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-26_00-45-14_S_1_E_2563_TS_335_0.00562827575170271_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-26_01-05-26_S_2_E_2563_TS_335_0.00562827575170271_rewards.npy")
    # all_performances_8 = np.array([perf1, perf2, perf3])
    # data8 = ("rewards", all_performances_8, "Ep: 2563, TS: 335, lr: 5.6e-3", 2563, 335, 0.00562827575170271)

    # perf1 = np.load("perf_metrics/2024-05-26_01-33-31_S_0_E_3066_TS_460_3.048598761934543e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-26_01-59-05_S_1_E_3066_TS_460_3.048598761934543e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-26_02-23-16_S_2_E_3066_TS_460_3.048598761934543e-05_rewards.npy")
    # all_performances_9 = np.array([perf1, perf2, perf3])
    # data9 = ("rewards", all_performances_9, "Ep: 3066, TS: 460, lr: 3.0e-5", 3066, 460, 3.048598761934543e-05)

    # perf1 = np.load("perf_metrics/2024-05-26_02-34-58_S_0_E_1947_TS_385_0.0023306699268543242_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-26_02-46-43_S_1_E_1947_TS_385_0.0023306699268543242_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-26_02-58-33_S_2_E_1947_TS_385_0.0023306699268543242_rewards.npy")
    # all_performances_10 = np.array([perf1, perf2, perf3])
    # data10 = ("rewards", all_performances_10, "Ep: 1947, TS: 385, lr: 2.3e-3", 1947, 385, 0.0023306699268543242)

    # perf1 = np.load("perf_metrics/2024-05-27_01-02-16_S_0_E_2553_TS_299_3.706288089267362e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_01-14-07_S_1_E_2553_TS_299_3.706288089267362e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_01-25-56_S_2_E_2553_TS_299_3.706288089267362e-05_rewards.npy")
    # all_performances_11 = np.array([perf1, perf2, perf3])
    # data11 = ("rewards", all_performances_11, "Ep: 2553, TS: 299, lr: 3.7e-5", 2553, 299, 3.706288089267362e-05)

    # perf1 = np.load("perf_metrics/2024-05-27_01-36-38_S_0_E_2897_TS_238_1.8658908758909327e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_01-47-16_S_1_E_2897_TS_238_1.8658908758909327e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_01-57-53_S_2_E_2897_TS_238_1.8658908758909327e-05_rewards.npy")
    # all_performances_12 = np.array([perf1, perf2, perf3])
    # data12 = ("rewards", all_performances_12, "Ep: 2897, TS: 238, lr: 1.9e-5", 2897, 238, 1.8658908758909327e-05)

    # perf1 = np.load("perf_metrics/2024-05-27_02-09-45_S_0_E_2119_TS_361_2.8247780768359425e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_02-21-29_S_1_E_2119_TS_361_2.8247780768359425e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_02-33-21_S_2_E_2119_TS_361_2.8247780768359425e-05_rewards.npy")
    # all_performances_13 = np.array([perf1, perf2, perf3])
    # data13 = ("rewards", all_performances_13, "Ep: 2119, TS: 361, lr: 2.8e-5", 2119, 361, 2.8247780768359425e-05)

    # perf1 = np.load("perf_metrics/2024-05-27_02-45-59_S_0_E_2503_TS_323_2.3394011077172677e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_02-58-45_S_1_E_2503_TS_323_2.3394011077172677e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_03-11-22_S_2_E_2503_TS_323_2.3394011077172677e-05_rewards.npy")
    # all_performances_14 = np.array([perf1, perf2, perf3])
    # data14 = ("rewards", all_performances_14, "Ep: 2503, TS: 323, lr: 2.3e-5", 2503, 323, 2.3394011077172677e-05)

    # perf1 = np.load("perf_metrics/2024-05-27_03-23-21_S_0_E_2581_TS_294_3.0333542713599293e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_03-35-23_S_1_E_2581_TS_294_3.0333542713599293e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_03-47-16_S_2_E_2581_TS_294_3.0333542713599293e-05_rewards.npy")
    # all_performances_15 = np.array([perf1, perf2, perf3])
    # data15 = ("rewards", all_performances_15, "Ep: 2581, TS: 294, lr: 3.0e-5", 2581, 294, 3.0333542713599293e-05)

    # perf1 = np.load("perf_metrics/2024-05-27_03-57-27_S_0_E_2733_TS_240_5.6257146931580896e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_04-07-36_S_1_E_2733_TS_240_5.6257146931580896e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_04-17-44_S_2_E_2733_TS_240_5.6257146931580896e-05_rewards.npy")
    # all_performances_16 = np.array([perf1, perf2, perf3])
    # data16 = ("rewards", all_performances_16, "Ep: 2733, TS: 240, lr: 5.6e-5", 2733, 240, 5.6257146931580896e-05)

    # perf1 = np.load("perf_metrics/2024-05-27_04-30-15_S_0_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_04-42-46_S_1_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_04-55-15_S_2_E_2711_TS_311_0.0002515289269330568_rewards.npy")
    # all_performances_17 = np.array([perf1, perf2, perf3])
    # data17 = ("rewards", all_performances_17, "Ep: 2711, TS: 311, lr: 2.5e-4", 2711, 311, 0.0002515289269330568)

    # perf1 = np.load("perf_metrics/2024-05-27_05-07-52_S_0_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_05-20-32_S_1_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_05-33-10_S_2_E_2469_TS_336_2.079508981715035e-05_rewards.npy")
    # all_performances_18 = np.array([perf1, perf2, perf3])
    # data18 = ("rewards", all_performances_18, "Ep: 2469, TS: 336, lr: 2.1e-5", 2469, 336, 2.079508981715035e-05)

    # perf1 = np.load("perf_metrics/2024-05-27_05-49-10_S_0_E_2872_TS_366_1.6280798813484343e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_06-05-11_S_1_E_2872_TS_366_1.6280798813484343e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_06-21-12_S_2_E_2872_TS_366_1.6280798813484343e-05_rewards.npy")
    # all_performances_19 = np.array([perf1, perf2, perf3])
    # data19 = ("rewards", all_performances_19, "Ep: 2872, TS: 366, lr: 1.6e-5", 2872, 366, 1.628079818348434e-05)

    # perf1 = np.load("perf_metrics/2024-05-27_06-33-58_S_0_E_2288_TS_376_1.0544223184265175e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/2024-05-27_06-46-43_S_1_E_2288_TS_376_1.0544223184265175e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/2024-05-27_06-59-33_S_2_E_2288_TS_376_1.0544223184265175e-05_rewards.npy")
    # all_performances_20 = np.array([perf1, perf2, perf3])
    # data20 = ("rewards", all_performances_20, "Ep: 2288, TS: 376, lr: 1.1e-5", 2288, 376, 1.0544223184265175e-05)

    # perf1 = np.load("perf_metrics/pendulum12_S_4_E_10000_TS_320_0.00025_rewards.npy")
    # perf2 = np.load("perf_metrics/pendulum12_S_9_E_10000_TS_320_0.00025_rewards.npy")
    # perf3 = np.load("perf_metrics/pendulum12_S_11_E_10000_TS_320_0.00025_rewards.npy")
    # all_performances_21 = np.array([perf1, perf2, perf3])
    # data21 = ("rewards", all_performances_21, "Ep: 10000, TS: 320, lr: 2.5e-4", 10000, 320, 0.00025)

    # perf1 = np.load("perf_metrics/old perf metrics/pendulum7_S_4_E_20000_TS_400_5e-05_rewards.npy")
    # perf2 = np.load("perf_metrics/old perf metrics/pendulum7_S_9_E_20000_TS_400_5e-05_rewards.npy")
    # perf3 = np.load("perf_metrics/old perf metrics/pendulum7_S_11_E_20000_TS_400_5e-05_rewards.npy")
    # all_performances22 = np.array([perf1, perf2, perf3])
    # data22 = ("rewards", all_performances22, "Ep: 20000, TS: 400, lr: 5e-5", 20000, 400, 5e-05)

    # perf1 = np.load("perf_metrics/old perf metrics/pendulum1_S_4_E_8000_TS_400_0.0001_rewards.npy")
    # perf2 = np.load("perf_metrics/old perf metrics/pendulum1_S_9_E_8000_TS_400_0.0001_rewards.npy")
    # perf3 = np.load("perf_metrics/old perf metrics/pendulum1_S_11_E_8000_TS_400_0.0001_rewards.npy")
    # all_performances23 = np.array([perf1, perf2, perf3])
    # data23 = ("rewards", all_performances23, "Ep: 8000, TS: 400, lr: 1e-4", 8000, 400, 0.0001)


    # all_performances = [data21, data1, data2, data22, data3, data23, data6, data8, data9, data11, data14, data18, data19, data20]
    # plot_performances_multiple(all_performances)
