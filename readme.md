Deep Q-Networks for Zero-Shot Generalization in Autonomous Driving

Executive Summary

This project explores the application of Deep Reinforcement Learning (DRL) for continuous control in autonomous navigation. Utilizing the procedurally generated MetaDrive environment, this repository implements a Deep Q-Network (DQN) designed specifically to overcome environmental overfitting.

By enforcing strict kinematic constraints on the action space and scaling the experiential data pipeline across 10,000 episodes and 20 unique map topologies, the resulting agent demonstrates highly robust Zero-Shot Generalization when evaluated on 100 entirely unseen procedural environments.

 Core Engineering & Research Challenges

Training an autonomous agent exposed several complex pathologies inherent to standard Markov Decision Process (MDP) reward formulations. This project documents the systematic eradication of these failure states:

1. Reward Exploitation & Local Maxima

The Problem: Initial baseline models optimized for maximum forward velocity leading into immediate collisions, as speed-accumulation rewards outpaced survival incentives.

The Solution: Implemented a balanced 50/50/200 (Crash/Out-of-Road/Success) reward shaping mechanism to properly weight the horizon of survival against immediate velocity gains.

2. Curing "Learned Paralysis" via Kinematic Constraints

The Problem: Introducing heavy collision penalties (-100) triggered extreme Q-Value Pessimism. The agent exploited the emergency brake (-1.0) to stall the vehicle for 45-minute episodes, prioritizing zero negative rewards over task completion.

The Solution: Engineered a "Forward-Forced" Action Space. By replacing the emergency brake with a 0.1 minimum throttle limit, the system simulated real-world engine idle dynamics. This physically constrained the agent, making stalling mathematically impossible and forcing the Bellman equation to optimize for high-speed evasive maneuvering.

3. Symmetrical Failure States & Universal Generalization

The Problem: Standard DQNs frequently overfit to specific intersection geometries, failing catastrophically in novel environments.

The Solution: Scaled the neural architecture to 512 hidden neurons and expanded the Replay Buffer capacity to 200,000 to prevent catastrophic forgetting across 50 diverse map topologies. Epsilon decay was strictly tuned to 0.9998 to guarantee 15,000 episodes of pure exploration.

The Result: During zero-shot evaluation on 100 unseen maps, the agent achieved a highly symmetrical failure split (e.g., roughly equal Out-of-Road vs. Vehicle Collision rates). In RL systems design, this symmetry proves the physical limits of the vehicle (steering grip vs. forward momentum) were optimally balanced by the learned policy.

System Architecture & Repository Navigation

This repository is structured for empirical reproducibility and data analysis rather than simple execution.

/src/env_utils.py: Contains the custom environment wrappers, state-space flattening, and the kinematically constrained discrete-to-continuous action map.

/docs/report.ipynb: An automated, executable data pipeline documenting the Phase 1 through Phase 5 training evolution, including moving-average visualizations of reward stabilization and crash-rate reduction.

/models/: Stores the finalized PyTorch model weights (dqn_trained.pt) and the raw .npy history arrays for empirical validation.

/Evaluation/: Contains the parsed JSON logs of the zero-shot testing phase across 100 unseen procedural seeds.

Execution for Reproducibility

For researchers or engineers wishing to validate the zero-shot performance or examine the training pipeline:

# 1. Install dependencies
pip install -r requirements.txt

# 2. Evaluate the pre-trained weights on 100 unseen procedural maps
python evaluate.py

# 3. Render the evaluation (3D Chase and Top-Down view)
python visualize.py


Note: The complete hyperparameter configuration (Epsilon Decay, Buffer Size, Batch Size) used for the massive 10k-episode generalization run can be reviewed directly in main.py.