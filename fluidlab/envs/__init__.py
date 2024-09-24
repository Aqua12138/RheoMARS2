
from gymnasium import register

for env_name in ['LatteArt', 'LatteArtStir', 'Scooping', 'GatheringEasy', 'GatheringO', 'IceCreamDynamic', 'IceCreamStatic', 'Transporting', 'Stabilizing', 'Pouring', 'Circulation', 'Mixing', 'Gathering', 'Reach', 'Accumulation']:
    for id in range(1):
        register(
            id = f'{env_name}-v{id}',
            entry_point=f'fluidlab.envs.{env_name.lower()}_env:{env_name}Env',
            max_episode_steps=10000
        )