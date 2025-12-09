import subprocess
import random
import time
import os
import signal

def launch_gazebo(world_path):
    print("launching gazebo")
    try:
        # Command to launch the Gazebo world using ROS 2
        launch_command = [
            'ros2', 'launch', 'gazebo_ros', 'gazebo.launch.py',
            f'world:={world_path}'
        ]

        # Start the process
        process = subprocess.Popen(launch_command)
        print(f"Launching: {world_path}")
        return process
    except Exception as e:
        print(f"Failed to launch Gazebo: {e}")

def kill_gazebo(process):
    try:
        # Send SIGINT to gracefully close the Gazebo process
        os.kill(process.pid, signal.SIGINT)
        print("Gazebo simulation stopped.")
    except Exception as e:
        print(f"Failed to stop Gazebo: {e}")

if __name__ == "__main__":
    # List of available stages
    worlds = [
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage1/burger.model",
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage2/burger.model",
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage3/burger.model",
        "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage4/burger.model",
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage5/burger.model",
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_stage6/burger.model",
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_open_world_1/burger.model",
        # "/home/fabio/ros2_humble/src/turtlebot3_simulations/turtlebot3_gazebo/worlds/turtlebot3_dqn_open_world_3/burger.model",
    ]

    while True:
        # Select a random world
        world_path = random.choice(worlds)

        # Launch the Gazebo simulation
        process = launch_gazebo(world_path)

        # Wait for 10 seconds
        time.sleep(10)

        # Kill the running Gazebo process
        kill_gazebo(process)

        # Wait a moment before launching the next simulation
        time.sleep(10)
