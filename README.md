# Deep Reinforcement Learning for TurtleBot3 Navigation

Training Deep Reinforcement Learning algorithms in the TurtleBot3 navigation task using ROS2. We provide:
- Multiple environment setups on Gazebo
- Implemented DRL algorithms - Dreamerv3, SAC, DDPG, TD3
<br>

| Stage 6 | Stage 4 |
| --- | --- |
| <img src="https://github.com/raulsteinmetz/turtlebot3-drl/assets/85199336/9f881aac-f87b-4b63-a323-655b47e3a18f" width="200"/> | <img src="https://github.com/raulsteinmetz/turtlebot3-drl/assets/85199336/007d7844-ebdb-47c3-b318-69e393d3c91d" width="200"/> |


<br>

## Configuring ROS packages
Refer to `./turtlebot3_gazebo/README.md`
  
## Training Your Agent
Refer to `./TRAIN.md`

## Folder Structure
- `./best_models` contains train and test information from algorithms and trained models
- `./dreamerv3-torch` holds code for the TurtleDreamer algorithm (Dreamerv3 for TurtleBot3)
- `./model_free` holds the implementations of sac, ddpg and td3
- `./models` keeps models and logs while the algorithms are training
- `./plots` keeps comparisson plots in pdf format
-  `./turtle_env` keeps the gym code for the TurtleBot3 task
-  `./turtlebot3_gazebo` keeps the ROS2 files needed for configuration  

## The code on this repository was inspired by:

- [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- [danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code)
- [dranaju/project](https://github.com/dranaju/project)
