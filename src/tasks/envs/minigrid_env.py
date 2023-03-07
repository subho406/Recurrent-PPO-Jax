import gymnasium as gym
import numpy as np
import time

from gymnasium.wrappers import TransformObservation
from gymnasium import spaces
from minigrid.wrappers import RGBImgPartialObsWrapper,ImgObsWrapper,ObservationWrapper,OneHotPartialObsWrapper

class ViewSizeWrapper(ObservationWrapper):
    """
    Wrapper to customize the agent field of view size.
    This cannot be used with fully observable wrappers.
    """

    def __init__(self, env, agent_view_size=7):
        super().__init__(env)

        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3

        self.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        new_image_space = gym.spaces.Box(
            low=0, high=255, shape=(agent_view_size, agent_view_size, 3), dtype="uint8"
        )

        # Override the environment's observation spaceexit
        self.observation_space = spaces.Dict(
            {**self.observation_space.spaces, "image": new_image_space}
        )

    def observation(self, obs):
        env = self.unwrapped

        grid, vis_mask = env.gen_obs_grid(self.agent_view_size)

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)
        return {**obs, "image": image}

def create_minigrid_env_pixel(**kwargs):
    if "Memory" in kwargs['name']:
        env=gym.make(kwargs['name'],render_mode='rgb_array',agent_view_size=kwargs.get('view_size', 3))
        tile_size = kwargs.get('tile_size', 28)
        env.action_space = spaces.Discrete(3)
        env = RGBImgPartialObsWrapper(env, tile_size = tile_size,) 
    else:
        env=gym.make(kwargs['name'],render_mode='rgb_array',agent_view_size=kwargs.get('view_size', 7))
        tile_size = kwargs.get('tile_size', 8)
        env = RGBImgPartialObsWrapper(env, tile_size = tile_size)
    env = ImgObsWrapper(env)
    env = TransformObservation(env, lambda obs: obs.astype(np.float32) / 255.)
    env.observation_space=spaces.Box(
            low = 0,
            high = 1.0,
            shape =  env.observation_space.shape,
            dtype = np.float32)
    return env


def create_minigrid_env_onehot(**kwargs):
    env=gym.make(kwargs['name'],render_mode='rgb_array')
    if "Memory" in kwargs['name']:
            view_size =  kwargs.get('view_size', 7)
            tile_size = kwargs.get('tile_size', 28)
            env.action_space = spaces.Discrete(3)
            env = ViewSizeWrapper(env, agent_view_size=view_size)
            
    else:
        view_size = kwargs.get('view_size', 7)
        tile_size = kwargs.get('tile_size', 8)
        env = ViewSizeWrapper(env, view_size)
    env = OneHotPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    env = TransformObservation(env, lambda obs: obs.astype(np.float32) )
    env.observation_space=spaces.Box(
            low = 0,
            high = 1.0,
            shape =  env.observation_space.shape,
            dtype = np.float32)
    return env
