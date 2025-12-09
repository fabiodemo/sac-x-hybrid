# Training Your Agent

This guide provides instructions on how to train the reinforcement learning agents in the TurtleBot3 simulation environment.

## 1. Launch the Gazebo Simulation

First, you need to launch the Gazebo simulation with the desired stage. The stage defines the world (i.e., the map and obstacle layout) for the TurtleBot3.

Open a new terminal and run the following command:

```bash
ros2 launch turtlebot3_gazebo turtle_stage<number>.py
```

Replace `<number>` with the stage number you want to use (e.g., 1, 2, 3, 4, etc.).

**Example:**
```bash
ros2 launch turtlebot3_gazebo turtle_stage1.py
```

## 2. Run the Training Script

Once the Gazebo simulation is running, you can start the training process. You can use the provided `train.sh` script, which automates the training, saving, and testing process.

Open another terminal and run the `train.sh` script with the desired arguments:

```bash
./train.sh <agent> <stage> <lidar_points> [load_models]
```

### Script Arguments

-   `<agent>`: The name of the reinforcement learning agent to train.
    -   **Options:** `sac`, `ddpg`, `td3`, `sac_x`, `sac_x_hybrid`
    -   **Default:** `ddpg`

-   `<stage>`: The environment stage number. This should match the stage you launched in Gazebo.
    -   **Options:** 1, 2, 3, 4, etc.
    -   **Default:** `1`

-   `<lidar_points>`: The number of LIDAR readings to use as input for the agent.
    -   **Options:** e.g., 10, 360
    -   **Default:** `10`

-   `[load_models]`: (Optional) Set to `True` to load and continue training from a previously saved model.
    -   **Options:** `True`, `False`
    -   **Default:** `False`

### Training Examples

**Example 1: Train the SAC-X-Hybrid agent on Stage 1 with 10 LIDAR points.**
```bash
./train.sh sac_x_hybrid 1 10
```

**Example 2: Train the DDPG agent on Stage 4 with 360 LIDAR points.**
```bash
./train.sh ddpg 4 360
```

**Example 3: Load and continue training a SAC agent on Stage 2.**
```bash
./train.sh sac 2 10 True
```

## 3. The Training Process (`train.sh`)

The `train.sh` script will:
1.  **Train the agent:** It calls `python train.py` with the arguments you provided.
2.  **Save the best model:** After training is complete, it runs `python save_to_best.py` to copy the best performing model to the `./best_models` directory.
3.  **Test the agent:** It runs `python test.py` to evaluate the final trained agent and save the test results.

## 4. Monitoring Training

You can monitor the training progress by looking at the `train.csv` file located in the agent's checkpoint directory (e.g., `models/<agent>/stage<stage>/`). This file contains the accumulated reward for each episode.

You can also use the `plt.py` script to plot the learning curve from this file.
```bash
python plt.py --agent <agent> --stage <stage> --lidar <lidar>
```