from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, Tuple

import chex
import jax
from jax import numpy as jnp

from .base import BaseEnv, BaseEnvParams, BaseEnvState


@dataclass
class AVEnv(BaseEnv):
    @partial(jax.jit, static_argnames=("env_params", "dt"))
    def step(
        key: chex.PRNGKey,
        env_state: BaseEnvState,
        action: chex.Scalar | chex.Array,
        env_params: BaseEnvParams,
        dt: float,
    ) -> Tuple[
        chex.Array, BaseEnvState, chex.Scalar | chex.Array, chex.Array, Dict[Any, Any]
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
        new_agent_pos = env_state.agent_pos + action * dt
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

        new_state = BaseEnvState(
            time=new_time,
            goal_pos=env_state.goal_pos,
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

    @partial(jax.jit, static_argnames=("env_params", "dt"))
    def _move_kinematic_obstacles(
        env_state: BaseEnvState, env_params: BaseEnvParams, dt: float
    ):
        positions = env_state.kinematic_obstacles
        velocities = env_state.kinematic_obst_velocities
        new_positions = positions + dt * velocities

        new_positions = jax.vmap(lambda x, y: jnp.mod(x, y), in_axes=(0, None))(
            new_positions,
            jnp.flip(jnp.asarray(env_params.map_height_width, dtype=jnp.int32)),
        )

        return new_positions
