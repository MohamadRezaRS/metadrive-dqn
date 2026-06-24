import numpy as np
import torch
from src.env_utils import create_env
from src.agent import DQNAgent

def train_agent():
    # 1. initialize environment and optimized hyperparameters
    env = create_env()
    num_episodes = 1000
    batch_size = 128
    gamma = 0.99
    sync_target_freq = 5  # sync target network every 5 episodes for faster tracking
    
    # epsilon scheduling tailored for 1000 episodes
    epsilon = 1.0
    epsilon_decay = 0.996
    min_epsilon = 0.10
    
    # 2. initialize agent
    
    state_shape = env.observation_space.shape[0]
    action_shape = env.action_space.n
    agent = DQNAgent(state_shape, action_shape, gamma=gamma)
    
    # 3. tracking Arrays for Task 8 Reports
    rewards_history = []
    steps_history = []
    crash_history = []  # 0 = success, 1 = out_of_road, 2 = crash_vehicle
    

    for episode in range(1, num_episodes + 1):
        state, info = env.reset()
        total_reward = 0
        steps = 0
        
        while True:
            # select action using epsilon-greedy strategy
            action = agent.select_action(state, epsilon)
            
            # step environment
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # store experience in replay buffer
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # train the network if buffer has enough data
            if len(agent.replay_buffer) > batch_size:
                agent.update_policy_network(batch_size)
                
            state = next_state
            total_reward += reward
            steps += 1
            
            if done:
                break
                
        # determine specific termination reason and map to numerical codes
        if info.get("arrive_dest", False):
            reason = "success_finish"
            crash_code = 0
        elif info.get("out_of_road", False):
            reason = "out_of_road"
            crash_code = 1
        else:
            reason = "crash_vehicle"
            crash_code = 2
            
        # append metrics to tracking lists
        rewards_history.append(total_reward)
        steps_history.append(steps)
        crash_history.append(crash_code)
        
        # sync target network
        if episode % sync_target_freq == 0:
            agent.update_target_network()
            
        # decay exploration rate
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        print(f"Episode: {episode:4d}/{num_episodes} | Steps: {steps:3d} | "
              f"Reward: {total_reward:6.2f} | Epsilon: {epsilon:.3f} | Reason: {reason}")

    # 4. save everything 
    torch.save(agent.policy_net.state_dict(), "models/dqn_trained.pt")
    np.save("models/rewards_history.npy", np.array(rewards_history))
    np.save("models/steps_history.npy", np.array(steps_history))
    np.save("models/crash_history.npy", np.array(crash_history))
    
    env.close()


if __name__ == "__main__":
    train_agent()