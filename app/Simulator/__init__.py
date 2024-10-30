from .sumo_env.SUMO import SUMO
from .carla_env.CARLA import CARLA

def make(env_name):
    if env_name == 'SUMO':
        return SUMO()
    elif env_name == 'CARLA':
        return CARLA()
    else:
        raise ValueError(f"Unknown environment: {env_name}")
