import jax

from env.autonomous_veh import AVEnv
from utils.autoreset import AVAutoResetWrapper
from utils.renderer import PygameFrontend

key = jax.random.PRNGKey(0)
env_params, init_state = AVEnv.init_params(
    key=key,
    map_id=1,
    max_steps=1000,
    discretization_scale=1,
    path_length=100,
    fps=60,
    perception_radius=5.0,
    num_ray_sensors=128,
)

env = AVAutoResetWrapper(AVEnv, env_params, init_state)

# TODO: test parallel envs

# test renderer
frontend = PygameFrontend(env, env_params, init_state, eval_mode=False)

frontend.run()
