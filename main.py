import numpy as np
import torch
import argparse

from src.env_utils import create_env, flatten_obs, discrete_to_continuous_action
from src.replay_buffer import ReplayBuffer
from src.agent import DQNAgent

def train_agent(num_episodes=10000, start_seed=1, num_scenarios=20):
    # 1. setup Environment
    env = create_env(start_seed=start_seed, num_scenarios=num_scenarios)
    
    # 2. optimized Hyperparameters
    batch_size = 256 
    gamma = 0.995 # increased discount factor to prioritize long-term rewards
    hidden_size = 512  # Increased hidden layer size
    sync_target_freq = 40  # Sync target network every 40 episodes
    
    # Epsilon scheduling tailored for 10000 episodes
    epsilon = 1.0
    epsilon_decay = 0.9995
    min_epsilon = 0.05
    
    # 3. initialize Agent and Custom Memory Bank
    state_size = 259 
    action_size = 6
    memory = ReplayBuffer(capacity=100000)  # Safe capacity for 10000 episodes
    agent = DQNAgent(state_size, hidden_size, action_size, gamma=gamma)
    
    # 4. tracking Arrays for  Reports
    rewards_history = []
    steps_history = []
    crash_history = []  # 0 = success, 1 = out_of_road, 2 = crash_vehicle
    
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = flatten_obs(obs)
        
        total_reward = 0
        steps = 0
        
        while True:
            # Step A: Select action index using epsilon-greedy strategy
            action_idx = agent.act(state, epsilon)
            
            # Step B: Translate discrete index to continuous MetaDrive controls
            continuous_action = discrete_to_continuous_action(action_idx)
            
            # Step C: Step environment
            next_obs, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated
            next_state = flatten_obs(next_obs)
            
            # Step D: Store experience in replay buffer
            memory.add(state, action_idx, reward, next_state, done)
            
            # Step E: Train the network if buffer has enough data
            if len(memory) > batch_size:
                batch = memory.sample(batch_size)
                agent.learn(batch)
                
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
                
        # Determine specific termination reason and map to numerical codes
        if info.get("arrive_dest", False):
            reason = "success_finish"
            crash_code = 0
        elif info.get("out_of_road", False):
            reason = "out_of_road"
            crash_code = 1
        else:
            reason = "crash_vehicle"
            crash_code = 2
            
        # Append metrics to tracking lists
        rewards_history.append(total_reward)
        steps_history.append(steps)
        crash_history.append(crash_code)
        
        # Sync target network
        if episode % sync_target_freq == 0:
            agent.update_target_network()
            
        # Decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        print(f"Episode: {episode:4d}/{num_episodes} | Steps: {steps:3d} | "
              f"Reward: {total_reward:6.2f} | Epsilon: {epsilon:.3f} | Reason: {reason}")

    # 5. save Everything at the Finish Line
    torch.save(agent.policy_net.state_dict(), "models/dqn_trained.pt")
    np.save("models/rewards_history.npy", np.array(rewards_history))
    np.save("models/steps_history.npy", np.array(steps_history))
    np.save("models/crash_history.npy", np.array(crash_history))
    
    env.close()



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the DQN agent on MetaDrive")
    parser.add_argument("--episodes", type=int, default=10000, help="Number of training episodes")
    parser.add_argument("--seed", type=int, default=1, help="starting map seed")
    parser.add_argument("--scenarios", type=int, default=20, help="number of different maps to train on")
    args = parser.parse_args()
    
    
    train_agent(args.episodes, args.seed, args.scenarios)