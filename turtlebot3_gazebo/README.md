# ROS2 Gazebo Environment Setup

This guide explains how to configure your ROS2 environment to use the custom Gazebo worlds and launch files required for this project.

## Prerequisites

1.  **Install ROS2:** You need a ROS2 distribution installed on your system. This project is tested with Foxy and Humble.
    -   [ROS2 Foxy Installation Guide](https://docs.ros.org/en/foxy/Installation/Ubuntu-Install-Debians.html)
    -   [ROS2 Humble Installation Guide](https://docs.ros.org/en/humble/Installation.html)

2.  **Install TurtleBot3 Simulation:** You must have the official TurtleBot3 simulation packages installed.
    -   [ROBOTIS e-Manual for TurtleBot3 Simulation](https://emanual.robotis.com/docs/en/platform/turtlebot3/simulation/)

## Configuration Steps

These steps involve replacing the default TurtleBot3 simulation's `launch`, `models`, and `worlds` files with the ones provided in this repository.

### 1. Set Your ROS2 Workspace Path

For convenience, let's set an environment variable for your ROS2 workspace directory. Replace `~/ros2_ws` with the actual path to your workspace.

```bash
export ROS2_WS=~/ros2_ws
```

### 2. Back Up Existing Files

Before copying the new files, it's a good practice to back up the original `launch`, `models`, and `worlds` directories from the `turtlebot3_gazebo` package.

```bash
# Create a backup directory if it doesn't exist
mkdir -p ~/turtlebot3_backup

# Move the original directories to the backup folder
mv $ROS2_WS/src/turtlebot3_simulations/turtlebot3_gazebo/launch ~/turtlebot3_backup/
mv $ROS2_WS/src/turtlebot3_simulations/turtlebot3_gazebo/models ~/turtlebot3_backup/
mv $ROS2_WS/src/turtlebot3_simulations/turtlebot3_gazebo/worlds ~/turtlebot3_backup/
```

### 3. Copy Project-Specific Files

Now, copy the `launch`, `models`, and `worlds` directories from this repository to your TurtleBot3 Gazebo package.

**Important:** These commands assume you are running them from the root of this (`sac-x-hybrid`) repository.

```bash
# Get the path to the turtlebot3_gazebo package in your workspace
TB3_GAZEBO_PATH="$ROS2_WS/src/turtlebot3_simulations/turtlebot3_gazebo"

# Copy the directories
cp -r ./turtlebot3_gazebo/launch $TB3_GAZEBO_PATH/
cp -r ./turtlebot3_gazebo/models $TB3_GAZEBO_PATH/
cp -r ./turtlebot3_gazebo/worlds $TB3_GAZEBO_PATH/
```

### 4. Build Your Workspace

Finally, build your ROS2 workspace to apply the changes.

```bash
cd $ROS2_WS
colcon build --symlink-install
```

After the build is complete, you will be able to launch the custom stages as described in the [TRAIN.md](../TRAIN.md) file.
