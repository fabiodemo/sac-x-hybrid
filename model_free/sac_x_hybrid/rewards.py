import numpy as np
import torch as T

REACH_TRESHOLD = 0.4
COLISION_TRESHOLD = 0.2

def reward_avoid_obstacles(lidar_data):
    if isinstance(lidar_data, T.Tensor):
        lidar_data = lidar_data.cpu().numpy()
    if lidar_data.ndim > 1:
        lidar_data = lidar_data.flatten()
    min_distance = np.min(lidar_data)
    safe_distance = 0.25
    if min_distance > safe_distance:
        return 0
    else:
        return -1 / min_distance

def reward_follow_walls(lidar_data, target_distance=0.25):
    current_distance = min(lidar_data)
    distance_error = abs(current_distance - target_distance)
    return -distance_error

def reward_efficient_exploration(visited_states, current_state):
    # This assumes current_state is either already a tuple or needs to be converted from a numpy array or list
    if not isinstance(current_state, tuple):
        state_tuple = tuple(current_state)
    else:
        state_tuple = current_state

    # Initialize the state's count in visited_states if it's not already there
    if state_tuple not in visited_states:
        visited_states[state_tuple] = 0

    # Increment the visit count for this state
    visited_states[state_tuple] += 1

    # Calculate the reward based on the logarithm of the visit count
    # The negative log function will decrease as visits increase, which is typical for exploration bonuses
    return -np.log(visited_states[state_tuple])

def get_reward_and_done(self, turtle_x, turtle_y, target_x, target_y, lidar_32):
    reward = 0
    done = False

    distance = np.sqrt((turtle_x - target_x)**2 + (turtle_y - target_y)**2)

    # Adicionando reward negativo proporcional à distância:
    # Se quiser "quanto mais perto, melhor", podemos subtrair a distância
    reward -= distance

    # Resto da sua lógica
    if distance < REACH_TRESHOLD:
        done = True
        reward += 100
    elif np.min(lidar_32) < COLISION_TRESHOLD:
        done = True
        reward -= 10
    elif self.step_counter >= (self.max_steps - 1):
        done = True
        reward -= 10
    
    reward += reward_efficient_exploration(self.visited_states, (turtle_x, turtle_y))

    return reward, done
