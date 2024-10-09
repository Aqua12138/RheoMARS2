
from gymnasium import register

for env_name in ['Shear', 'Compression', 'Flow']:
    for id in range(1):
        register(
            id = f'{env_name}-v{id}',
            entry_point=f'fluidlab.envs.{env_name.lower()}_env:{env_name}Env',
            max_episode_steps=10000
        )