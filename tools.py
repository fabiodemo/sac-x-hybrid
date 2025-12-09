from turtle_env.turtle import Turtle

from model_free.sac.sac_torch import Agent as SAC
from model_free.ddpg.ddpg_torch import Agent as DDPG
from model_free.td3.td3_torch import Agent as TD3
from model_free.sac_x_hybrid.sac_x_hybrid_torch import Agent as SACXH
from model_free.sac_x.sac_x_torch import Agent as SACX


def make_env(stage: int, max_steps: int, lidar: int):
    return Turtle(stage, max_steps, lidar)

def make_agent(env: Turtle, configs: dict, test: bool = True, shared_memory = None):
    if configs['agent'] == 'sac':
        agent = SAC(
            alpha=configs['alpha'],
            beta=configs['beta'],
            input_dims=env.observation_space.spaces['sensor_readings'].shape[0] + \
                env.observation_space.spaces['target'].shape[0] + \
                env.observation_space.spaces['velocity'].shape[0],
            max_action=1,
            gamma=configs['gamma'],
            n_actions=env.action_space.shape[0],
            max_size=configs['max_size'],
            tau=configs['tau'],
            batch_size=configs['batch_size'],
            reward_scale=configs['reward_scale'],
            min_action=-1, # checkout how this is used
            # checkpoint_dir = (
            #     f"{configs['checkpoint_dir']}/stage{configs['stage']}" if not test
            #     else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage{configs['stage']}"
            # )
            # checkpoint_dir = (
            #     f"{configs['checkpoint_dir']}/stage_random" if not test
            #     else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage_random"
            # ),
            shared_memory=shared_memory
            

        )
    elif configs['agent'] == 'ddpg':
        agent = DDPG(
            alpha=configs['alpha'],
            beta=configs['beta'],
            tau=configs['tau'],
            n_actions=env.action_space.shape[0],
            input_dims=env.observation_space.spaces['sensor_readings'].shape[0] + \
                env.observation_space.spaces['target'].shape[0] + \
                env.observation_space.spaces['velocity'].shape[0],
            gamma=configs['gamma'],
            max_size=configs['max_size'],
            fc1_dims=configs['fc1_dims'],
            fc2_dims=configs['fc2_dims'],
            batch_size=configs['batch_size'],
            max_action=1,
            min_action=-1, # checkout how this is used
            checkpoint_dir = (
                f"{configs['checkpoint_dir']}/stage_random" if not test
                else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage_random"
            )
        )
    elif configs['agent'] == 'td3':
        agent = TD3(
            alpha=configs['alpha'],
            beta=configs['beta'],
            input_dims=env.observation_space.spaces['sensor_readings'].shape[0] + \
                env.observation_space.spaces['target'].shape[0] + \
                env.observation_space.spaces['velocity'].shape[0],
            tau=configs['tau'],
            max_action=1,
            min_action=-1, # checkout how this is used
            gamma=configs['gamma'],
            update_actor_interval=configs['update_actor_interval'], # dpu
            warmup=configs['warmup'] if not configs['test'] else 0,
            n_actions=env.action_space.shape[0],
            max_size=configs['max_size'],
            layer1_size=configs['layer1_size'],
            layer2_size=configs['layer2_size'],
            batch_size=configs['batch_size'],
            noise=configs['noise'],
            checkpoint_dir = (
                f"{configs['checkpoint_dir']}/stage_random" if not test
                else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage_random"
            )
        )
    elif configs['agent'] == 'sac_x_hybrid':
        agent = SACXH(
            alpha=configs['alpha'],
            beta=configs['beta'],
            input_dims=env.observation_space.spaces['sensor_readings'].shape[0] + \
                env.observation_space.spaces['target'].shape[0] + \
                env.observation_space.spaces['velocity'].shape[0],
            max_action=1,
            gamma=configs['gamma'],
            n_actions=env.action_space.shape[0],
            max_size=configs['max_size'],
            tau=configs['tau'],
            batch_size=configs['batch_size'],
            reward_scale=configs['reward_scale'],
            min_action=-1, # checkout how this is used
            checkpoint_dir = (
                f"{configs['checkpoint_dir']}/stage{configs['stage']}" if not test
                else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage{configs['stage']}"
            )

,
            num_skills=configs['num_skills']
        )
    elif configs['agent'] == 'sac_x':
        agent = SACX(
            alpha=configs['alpha'],
            beta=configs['beta'],
            input_dims=env.observation_space.spaces['sensor_readings'].shape[0] + \
                env.observation_space.spaces['target'].shape[0] + \
                env.observation_space.spaces['velocity'].shape[0],
            max_action=1,
            gamma=configs['gamma'],
            n_actions=env.action_space.shape[0],
            max_size=configs['max_size'],
            tau=configs['tau'],
            batch_size=configs['batch_size'],
            reward_scale=configs['reward_scale'],
            min_action=-1, # checkout how this is used
            checkpoint_dir = (
                f"{configs['checkpoint_dir']}/stage_random" if not test
                else f"best_models/lidar{configs['lidar']}/{configs['agent']}/stage_random"
            ),
            num_skills=configs['num_skills']
        )
    else:
        raise('Agent not implemented')
    
    return agent