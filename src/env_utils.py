import numpy as np
from metadrive.envs.metadrive_env import MetaDriveEnv


def create_env(start_seed,num_scenarios):
    """
    sets up the MetaDrive game with the exact rules we want for training
    """
    config = {
        "num_scenarios": num_scenarios,  # number of different road layouts to train on
        "start_seed": start_seed,            
        "use_render": False,        
        "crash_vehicle_done": True, # end the game immediately if the car crashes into something
        "out_of_route_done": True,   # end the game immediately if the car drives off the road
        #  THE BALANCED REWARD SHAPING 
        "crash_vehicle_penalty": 50.0,  
        "out_of_road_penalty": 50.0,    
        "success_reward": 200.0,
    }
    return MetaDriveEnv(config)



def flatten_obs(obs):
    """
    translates the complex game data into a flat list of numbers for the AI
    """
    return np.array(obs, dtype=np.float32).flatten()


def discrete_to_continuous_action(action_idx):
    """
    translates the AI's simple choice (0-5) into real joystick movements [Steering, Gas/Brake].
    updated mapping provides smooth cornering and traction control to prevent spin-outs.
    """

    action_map = {
        0: [-1.0,  0.1], # 0: Hard Left + Engine Brake (Dodging)
        1: [-0.75, 0.6], # 1: Mild Left + Forward (High-speed cornering)
        2: [ 0.0,  0.1], # 2: Straight + Engine Brake (Slowing down for traffic, NO PARKING)
        3: [ 0.0,  0.8], # 3: Straight + Gas (Score accumulator)
        4: [ 1.0,  0.1], # 4: Hard Right + Engine Brake (Dodging)
        5: [ 0.75, 0.6]  # 5: Mild Right + Forward (High-speed cornering)
        }
    # return the mapped action. If something goes wrong, default to [0.0, 0.0] (do nothing)
    return np.array(action_map.get(action_idx, [0.0, 0.0]))