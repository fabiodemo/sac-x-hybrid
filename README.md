# Deep Reinforcement Learning for TurtleBot3 Navigation

This repository provides a ROS2-based framework for training and evaluating Deep Reinforcement Learning (DRL) algorithms for the navigation task of a TurtleBot3 robot in a Gazebo-simulated environment.

It features implementations of several DRL agents, with a main contribution of **SAC-X-Hybrid (SAC-XH)**, a streamlined variant of SAC-X for multi-objective learning.

Six training environments of increasing complexity were used, ranging from an empty room (Stage 1) to a cluttered open space with many obstacles (Stage 6). Stages 1–3 introduce static obstacles of varying shapes, while Stages 4–6 progressively increase obstacle density and layout difficulty.

![Environments](figures/environments.tiff)

## Getting Started

To get started with this project, follow these steps:

1.  **Configure Environment:** Set up your ROS2 and Gazebo environment by following the instructions in the [ROS2 Gazebo Environment Setup](./turtlebot3_gazebo/README.md) guide.

2.  **Train an Agent:** Learn how to train the DRL agents by following the [Training Your Agent](./TRAIN.md) guide.

## Folder Structure

-   `./configs.yaml`: Contains hyperparameters for all DRL agents.
-   `./train.py`: The main script for training agents.
-   `./test.py`: The main script for evaluating trained agents.
-   `./tools.py`: Helper functions for creating agents and environments.
-   `./model_free/`: Holds the implementations of the DRL algorithms (SAC, DDPG, TD3, SAC-X, and SAC-X-Hybrid).
-   `./models/`: Stores model checkpoints and logs during training.
-   `./best_models/`: Stores the best performing models, training logs, and test results.
-   `./plots/`: Stores comparison plots in PDF format.
-   `./turtle_env/`: Contains the OpenAI Gym-like environment for the TurtleBot3 task.
-   `./turtlebot3_gazebo/`: Contains the ROS2 files for the Gazebo simulation, including custom launch files and worlds.

## Main Contribution: SAC-X-Hybrid (SAC-XH)

This work presents the **SAC-XH**, a streamlined variant of SAC-X that removes the explicit meta-controller and trains all skills jointly through a shared replay buffer. The approach unifies auxiliary and primary rewards into a single training signal, enabling one policy and critic to learn from both objectives simultaneously. SAC-XH reduces computational overhead while retaining the exploratory benefits of SAC-X through an implicit scheduling mechanism, allowing the agent to balance exploration among objectives without a high-level controller. This formulation makes the method suitable for resource-constrained robotic platforms while maintaining multi-objective learning capability.

### Auxiliary Rewards
A limited set of auxiliary signals was introduced to guide learning toward practical navigation behaviors, including collision avoidance, wall-following, and area exploration. These signals provide additional learning targets during off-policy updates, increasing the occurrence of meaningful transitions under sparse main rewards. The combined reward at each timestep is given by:

$r_t = r_{\text{main}}(t) + \lambda_1 r_{\text{collision}}(t) + \lambda_2 r_{\text{wall}}(t) + \lambda_3 r_{\text{explore}}(t),$

where $r_{\text{main}}(t)$ denotes the goal-oriented term, and $(\lambda_1, \lambda_2, \lambda_3)$ are hyperparameters tuned via grid search.

The auxiliary functions are defined as:

$$r_{\text{collision}}(t) =
\begin{cases}
    -\dfrac{\lambda_c}{d_{\min}(t)}, & \text{if } d_{\min}(t) < d_{\text{safe}},\\[6pt]
    0, & \text{otherwise,}
\end{cases}$$

$r_{\text{wall}}(t) = -|d_{\min}(t) - d_{\text{target}}|,$

$r_{\text{explore}}(t) = -\log(n_{\text{visits}}(s_t)).$

### Optimization Objective
The SAC-XH objective extends the standard SAC loss by adding auxiliary components:

$$\mathcal{L}_{SAC\text{-}XH} = \mathcal{L}_{SAC} + \sum_{i=1}^{N} \lambda_i \, \mathcal{L}_{aux}^{(i)}.$$

In practice, this formulation led to navigation patterns characterized by stable motion near obstacles and consistent distance maintenance, showing that auxiliary rewards helped the policy maintain consistency under sparse rewards.

The Algorithm implements the core training procedure for SAC-XH. The method samples transitions from a prioritized replay buffer $\mathcal{D}$ and computes intrinsic rewards $\mathcal{R}_{\text{skill}}$ based on skill execution. These rewards guide the training of auxiliary skill policies alongside the main SAC components (value network, actor, and critics). The update sequence follows a fixed order: value estimation, policy improvement, critic updates via temporal difference learning, and target network synchronization. Skill policies receive independent gradient updates using their corresponding intrinsic rewards. This design allows the agent to learn both goal-directed behavior and reusable skills without predefined task hierarchies.

## Evaluation

To evaluate a trained agent, you can use the `test.py` script. The `train.sh` script runs this automatically after training, but you can also run it manually.

```bash
python test.py --agent <agent_name> --stage <stage_num> --lidar <lidar_points>
```

The test results, including a video of the agent's performance, will be saved in the corresponding directory under `./best_models/`.

## Training video
[![Video](https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ5Pij7-XYFT3w_Pry6bs6YUARQMBMty6ctfg&s)](https://youtu.be/fA-_Mi_iMZ4")


## Publication

This project implements the algorithms described in:

> Demo Rosa F, Steinmetz R, Tello Gamarra DF. **Hybrid Soft Actor-Critic with Curriculum Learning for Sparse-Reward Mobile Robot Navigation**. *Journal of Intelligent & Fuzzy Systems: Applications in Engineering and Technology*. 2026;0(0). doi:[10.1177/18758967261431354](https://doi.org/10.1177/18758967261431354)

## Acknowledgements

The code on this repository was inspired by:
- [NM512/dreamerv3-torch](https://github.com/NM512/dreamerv3-torch)
- [danijar/dreamerv3](https://github.com/danijar/dreamerv3)
- [philtabor/Actor-Critic-Methods-Paper-To-Code](https://github.com/philtabor/Actor-Critic-Methods-Paper-To-Code)
- [dranaju/project](https://github.com/dranaju/project)

This repository is a joint effort with:
- [raulsteinmetz/turtlebot-dreamerv3](https://github.com/raulsteinmetz/turtlebot-dreamerv3)
