from algorithms.ppo import PPOAgent
import gym
import torch

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())

# Print the name of the GPU
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

def test_train_agent():
    env = gym.make('Pendulum-v1')
    agent = PPOAgent(env)
    agent.train(episodes=10, max_timesteps=1000)
    print("Training completed")

def test_visualize_agent(path):
    env = gym.make('Pendulum-v1', render_mode="human")
    agent = PPOAgent(env)
    
    # Load the trained model parameters
    agent.load_model(path)
    
    # Visualize the agent's performance
    agent.visualize(max_timesteps=200)

if __name__ == "__main__":
    test_train_agent()
    # test_visualize_agent()