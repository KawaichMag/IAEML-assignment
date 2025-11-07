import jax

from env.base import BaseEnv
from utils.autoreset import AutoResetWrapper
from utils.renderer import PygameFrontend

key = jax.random.PRNGKey(0)
env_params, init_state = BaseEnv.init_params(
    key=key,
    map_id=1,
    max_steps=1000,
    discretization_scale=1,
    path_length=100,
    fps=60,
    perception_radius=5.0,
    num_ray_sensors=32,
)

env = AutoResetWrapper(BaseEnv, env_params, init_state)

# TODO: test parallel envs

# test renderer
frontend = PygameFrontend(env, env_params, init_state, eval_mode=False)

frontend.run()
