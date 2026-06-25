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
        "out_of_route_done": True   # end the game immediately if the car drives off the road
    }
    return MetaDriveEnv(config)



def flatten_obs(obs):
    """
    translates the complex game data into a flat list of numbers for the AI
    """
    return np.array(obs, dtype=np.float32).flatten()

import numpy as np

def discrete_to_continuous_action(action_idx):
    """
    translates the AI's simple choice (0-5) into real joystick movements [Steering, Gas/Brake].
    updated mapping provides smooth cornering and traction control to prevent spin-outs.
    """

    action_map = {
            0: [-0.8, -0.95], # 0: Left + Brake
            1: [-0.4,  0.7], # 1: Left + Forward
            2: [ 0.0, -1.0], # 2: Straight + Brake (Heavy emergency brake)
            3: [ 0.0,  0.8], # 3: Straight + Forward (Cruising throttle)
            4: [ 0.8, -0.95], # 4: Right + Brake
            5: [ 0.4,  0.7]  # 5: Right + Forward
        }
    # return the mapped action. If something goes wrong, default to [0.0, 0.0] (do nothing)
    return np.array(action_map.get(action_idx, [0.0, 0.0]))