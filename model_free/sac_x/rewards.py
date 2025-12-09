import numpy as np
import torch as T

def reward_avoid_obstacles(lidar_data):
    """
    Calculate the reward for avoiding obstacles based on LIDAR readings.
    The reward is zero if the minimum distance is greater than the safe distance,
    otherwise, it is inversely proportional to the minimum distance.

    Args:
        lidar_data: LIDAR sensor readings as a numpy array or torch tensor.

    Returns:
        float: Reward value for obstacle avoidance.
    """
    if isinstance(lidar_data, T.Tensor):
        lidar_data = lidar_data.cpu().numpy()  # Convert to numpy if it's a tensor
    lidar_data = np.array(lidar_data).flatten()  # Ensure it is a flat array
    min_distance = np.min(lidar_data)
    safe_distance = 0.25
    return 0 if min_distance > safe_distance else -1 / min_distance

def reward_follow_walls(lidar_data, target_distance=0.25):
    """
    Calculate the reward for maintaining a specific distance to walls.
    The reward is the negative squared difference from the target distance.

    Args:
        lidar_data: LIDAR sensor readings as a numpy array or torch tensor.
        target_distance: Target distance to maintain from a wall.

    Returns:
        float: Reward value for following walls.
    """
    lidar_data = np.array(lidar_data).flatten()  # Ensure it is a flat array
    current_distance = np.min(lidar_data)
    return -(current_distance - target_distance)**2

def reward_efficient_exploration(visited_states, current_state):
    """
    Encourage exploration by providing a negative log of visitation count as reward.
    This promotes visiting less frequented states.

    Args:
        visited_states: Dictionary keeping track of visit counts for each state.
        current_state: Current state of the agent, can be a list, tuple, or numpy array.

    Returns:
        float: Reward value for exploring new states.
    """
    state_tuple = tuple(current_state) if not isinstance(current_state, tuple) else current_state
    visited_states[state_tuple] = visited_states.get(state_tuple, 0) + 1
    return -np.log(visited_states[state_tuple])

def reward_target_distance(current_position, target_position, optimal_distance=1.0):
    """
    Calculate the reward based on the distance to a target position.
    The reward is the negative of the absolute difference from the optimal distance.

    Args:
        current_position: Current position of the agent, should be a numpy array.
        target_position: Target position, should be a numpy array.
        optimal_distance: Optimal distance to maintain from the target.

    Returns:
        float: Reward for maintaining the optimal distance to the target.
    """
    current_distance = np.linalg.norm(current_position - target_position)
    return -abs(current_distance - optimal_distance)