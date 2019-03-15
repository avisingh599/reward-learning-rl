from .adapters.gym_adapter import (
    GYM_ENVIRONMENTS,
    GymAdapter,
)

ENVIRONMENTS = {
    'gym': GYM_ENVIRONMENTS,
}

ADAPTERS = {
    'gym': GymAdapter,
}


def get_environment(universe, domain, task, env_params):
    return ADAPTERS[universe](domain=domain, task=task, **env_params)


def get_environment_from_variant(variant):
    universe = variant['universe']
    task = variant['task']
    domain = variant['domain']
    env_params = variant['env_params']

    return get_environment(universe, domain, task, env_params)

def get_goal_example_environment_from_variant(variant):
    import gym
    
    if variant['task'] not in [env.id for env  in gym.envs.registry.all()]:
        from multiworld.envs.mujoco import register_goal_example_envs
        register_goal_example_envs()

    return GymAdapter(env=gym.make(variant['task']))
