from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Tuple

import chex
import equinox
import jax
from jax import numpy as jnp

from env.tasks import BaseTaskSampler
from utils import utils

from .base import BaseEnv, BaseEnvParams, BaseEnvState

RADIUS = 1.0  # radius of each circle
NUM_RAY_SENSORS = 16
FOV = jnp.pi  # used to get 360 degree vision
OBSTACLE_COST = 1000  # we are using int16, so choose this carefully
FPS = 60

PATHFINDING_MOVES = jnp.array([[0, 1], [0, -1], [-1, 0], [1, 0]], dtype=jnp.float32)
PATHFINDING_MOVES_DIAG = jnp.array(
    [[1, 1], [-1, -1], [-1, 1], [1, -1]], dtype=jnp.float32
)


class AVEnvState(BaseEnvState):
    agent_velocity: chex.Array


class AVEnvParams(BaseEnvParams):
    frickness: float


@dataclass
class AVEnv(BaseEnv):
    def init_params(
        key: chex.PRNGKey,
        map_id: int,
        max_steps: int,
        path_length: int,
        discretization_scale: int = 1,
        perception_radius: float = 5,
        num_ray_sensors: float = NUM_RAY_SENSORS,
        fov: float = FOV,
        fps: float = FPS,
        frickness: float = 2,
    ) -> Tuple[AVEnvParams, BaseEnvState]:
        if discretization_scale != 1:
            raise Exception("discretization_scale != 1 is not implemented!")

        task = BaseTaskSampler(map_id)
        # TODO: get map height and width
        env_params = AVEnvParams(
            max_steps_in_episode=max_steps,
            fps=fps,
            step_size=1 / fps,
            map_height_width=(task.height, task.width),
            nav_grid_shape_dtype=(
                (task.height * discretization_scale, task.width * discretization_scale),
                jnp.float32,
            ),
            path_array_shape_dtype=((path_length, 2), jnp.float32),
            discretization_scale=discretization_scale,
            perception_radius=perception_radius,
            agent_FOV=fov,
            num_ray_sensors=num_ray_sensors,
            frickness=frickness,
        )

        shape, dtype = env_params.path_array_shape_dtype

        convert_to_world_f = partial(
            utils.convert_to_world_view, map_shape=env_params.map_height_width
        )
        convert_to_world_vmap = jax.vmap(convert_to_world_f, in_axes=(0))

        init_state = AVEnvState(
            time=jnp.asarray(0),
            goal_pos=convert_to_world_f(task.goal_pos).astype(jnp.float32),
            agent_pos=convert_to_world_f(task.agent_pos).astype(jnp.float32),
            agent_velocity=jnp.array([0, 0]).astype(jnp.float32),
            agent_forward_dir=task.agent_forward_dir.astype(jnp.float32),
            # agent_goal_dir=task.agent_goal_dir,
            static_obstacles=convert_to_world_vmap(task.static_obstacles).astype(
                jnp.float32
            ),
            kinematic_obstacles=convert_to_world_vmap(task.kinematic_obstacles).astype(
                jnp.float32
            ),
            kinematic_obst_velocities=task.kinematic_obst_velocities.astype(
                jnp.float32
            ),
            path_array=jnp.zeros(shape=shape, dtype=dtype),
        )
        obstacles = jnp.concatenate(
            [init_state.static_obstacles, init_state.kinematic_obstacles], axis=0
        )
        path_array = BaseEnv._find_path(
            init_state.agent_pos, init_state.goal_pos, obstacles, env_params
        )
        # replace path_array
        init_state = equinox.tree_at(lambda t: t.path_array, init_state, path_array)

        return env_params, init_state

    @partial(jax.jit, static_argnames=("env_params",))
    def step(
        key: chex.PRNGKey,
        env_state: AVEnvState,
        action: chex.Scalar | chex.Array,
        env_params: AVEnvParams,
        dt: jax.typing.ArrayLike,
    ) -> Tuple[
        chex.Array, AVEnvState, chex.Scalar | chex.Array, chex.Array, Dict[Any, Any]
    ]:
        # concatenate static and dynamic obstacles
        obstacles = jnp.concatenate(
            [env_state.static_obstacles, env_state.kinematic_obstacles], axis=0
        )

        # Check Done.
        goal_done = AVEnv._check_goal(env_state.agent_pos, env_state.goal_pos)
        collision_done = AVEnv._check_collisions(env_state.agent_pos, obstacles)
        time_done = env_state.time >= env_params.max_steps_in_episode

        done = jnp.logical_or(goal_done, jnp.logical_or(collision_done, time_done))

        # Calculate reward
        true_reward_f = lambda _: 1.0  # Define your own
        false_reward_f = lambda _: 0.0  # Define your own
        reward = jax.lax.cond(goal_done, true_reward_f, false_reward_f, None)
        # NOTE: what I have define above is called sparse reward. You dont have to use a sparse reward

        # Move agent
        velocity_direction = env_state.agent_velocity / (
            jnp.linalg.norm(env_state.agent_velocity) + 1e-6
        )

        new_agent_velocity = (
            env_state.agent_velocity
            + (action - velocity_direction * env_params.frickness) * dt
        )
        new_agent_pos = env_state.agent_pos + new_agent_velocity * dt
        kinematic_obstacles = AVEnv._move_kinematic_obstacles(env_state, env_params, dt)

        # Increment time
        new_time = env_state.time + 1

        # path array. update every second.
        pred = jnp.mod(new_time, env_params.fps)
        path_array = jax.lax.cond(
            pred,
            lambda _: AVEnv._find_path(
                new_agent_pos, env_state.goal_pos, obstacles, env_params
            ),
            lambda _: env_state.path_array,
            None,
        )

        # action without direction
        pred = jnp.linalg.norm(action) < 1e-8

        agent_forward_dir = jax.lax.cond(
            pred,
            lambda _: env_state.agent_forward_dir,
            lambda _: action / (jnp.linalg.norm(action) + 1e-6),
            None,
        )

        new_state = AVEnvState(
            time=new_time,
            goal_pos=env_state.goal_pos,
            agent_velocity=new_agent_velocity,
            agent_pos=new_agent_pos,
            agent_forward_dir=agent_forward_dir,
            # agent_goal_dir=env_state.agent_goal_dir,
            static_obstacles=env_state.static_obstacles,
            kinematic_obstacles=kinematic_obstacles,
            kinematic_obst_velocities=env_state.kinematic_obst_velocities,
            path_array=path_array,
        )
        obs = AVEnv.get_observation(new_state, env_params)
        info = {
            "time": new_time,
            "distance_to_path": obs.distance_to_path,
            "direction_of_path": obs.direction_of_path,
        }
        # NOTE: info is used for evaluation

        return obs, new_state, reward, done, info

    @partial(jax.jit, static_argnames=("env_params",))
    def _move_kinematic_obstacles(
        env_state: AVEnvState, env_params: BaseEnvParams, dt: jax.typing.ArrayLike
    ):
        positions = env_state.kinematic_obstacles
        velocities = env_state.kinematic_obst_velocities
        new_positions = positions + velocities * dt

        new_positions = jax.vmap(lambda x, y: jnp.mod(x, y), in_axes=(0, None))(
            new_positions,
            jnp.flip(jnp.asarray(env_params.map_height_width, dtype=jnp.int32)),
        )

        return new_positions
