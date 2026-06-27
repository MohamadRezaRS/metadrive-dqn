import argparse
import json
import numpy as np
import torch

from src.env_utils import create_env, flatten_obs, discrete_to_continuous_action
from src.agent import DQNAgent

def evaluate_agent(num_episodes, start_seed, num_scenarios):
    # initialize environment with custom map settings
    env = create_env(start_seed=start_seed, num_scenarios=num_scenarios)
    
    state_size = 259 
    action_size = 6
    hidden_size = 512
    agent = DQNAgent(state_size,hidden_size, action_size)
    

    agent.policy_net.load_state_dict(torch.load("models/dqn_trained.pt", map_location='cpu'))
    agent.policy_net.eval()
    

    total_rewards = []
    success_count = 0
    out_of_road_count = 0
    crash_vehicle_count = 0
    
    for episode in range(1, num_episodes + 1):
        obs, info = env.reset()
        state = flatten_obs(obs)
        episode_reward = 0
        steps = 0
        
        while True:
            action_idx = agent.act(state, epsilon=0.0)
            continuous_action = discrete_to_continuous_action(action_idx)
            
            next_obs, reward, terminated, truncated, info = env.step(continuous_action)
            done = terminated or truncated
            
            state = flatten_obs(next_obs)
            episode_reward += reward
            steps += 1
            
            if done:
                if info.get("arrive_dest", False):
                    success_count += 1
                    reason = "success"
                elif info.get("out_of_road", False):
                    out_of_road_count += 1
                    reason = "out_of_road"
                else:
                    crash_vehicle_count += 1
                    reason = "crash_vehicle"
                break
                
        total_rewards.append(episode_reward)
        print(f"eval episode {episode:2d}/{num_episodes} | steps: {steps:3d} | reward: {episode_reward:6.2f} | result: {reason}")

    # calculate final aggregates
    average_reward = float(np.mean(total_rewards))
    success_rate = (success_count / num_episodes) * 100
    out_of_road_rate = (out_of_road_count / num_episodes) * 100
    crash_vehicle_rate = (crash_vehicle_count / num_episodes) * 100


    print(f"average reward:      {average_reward:.2f}")
    print(f"success rate:        {success_rate:.1f}%")
    print(f"out of road rate:    {out_of_road_rate:.1f}%")
    print(f"vehicle crash rate:  {crash_vehicle_rate:.1f}%")
    
    
    # serialize metrics for report generation
    report_data = {
        "average_reward": average_reward,
        "success_rate": success_rate,
        "out_of_road_rate": out_of_road_rate,
        "crash_vehicle_rate": crash_vehicle_rate,
        "tested_scenarios": num_scenarios,
        "start_seed": start_seed
    }
    
    with open("Evaluation/evaluation_report.json", "w") as file:
        json.dump(report_data, file, indent=4)
        
    print("evaluation metrics saved to Evaluation/evaluation_report.json")
    
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="evaluate trained dqn agent across multiple maps")
    parser.add_argument("--episodes", type=int, default=100, help="number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=21, help="starting map seed")
    parser.add_argument("--scenarios", type=int, default=100, help="number of different maps to test")
    
    args = parser.parse_args()
    evaluate_agent(args.episodes, args.seed, args.scenarios)