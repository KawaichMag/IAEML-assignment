import math
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import pygame
from jax import random

from env.base import BaseEnvParams
from utils import utils
from utils.autoreset import AutoResetWrapper

FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
DARK_GRAY = (30, 30, 30)
RED = (200, 50, 50)
BLUE = (50, 50, 200)
GREEN = (50, 200, 50)
YELLOW = (200, 200, 50)

# TODO: draw time


@dataclass
class PygameFrontend:
    env: AutoResetWrapper
    params: BaseEnvParams
    init_state: any
    eval_mode: bool = False
    agent_fn: any = None

    info_position: tuple[int, int] = (10, 10)
    info_font_size: int = 15
    info_line_height: int = 25
    info_color: tuple[int, int, int] = (255, 255, 255)

    move_speed: int = 5

    alpha: int = 80

    dt: int = 0

    @property
    def path_array(self):
        return self.rotate_to_map_vmap(self.state.path_array)

    @property
    def static_obst_map(self):
        return self.rotate_to_map_vmap(self.state.static_obstacles)

    @property
    def kin_obst_map(self):
        return self.rotate_to_map_vmap(self.state.kinematic_obstacles)

    @property
    def agent_pos(self):
        return self.rotate_to_map_vmap(jnp.atleast_2d(self.state.agent_pos))[0]

    @property
    def goal_pos(self):
        return self.rotate_to_map_vmap(jnp.atleast_2d(self.state.goal_pos))[0]

    @property
    def agent_forward_dir(self):
        return jnp.flip(self.state.agent_forward_dir) * jnp.array([-1, 1])

    @property
    def rays(self):
        return jnp.flip(self.obs.collision_rays, axis=-1) * jnp.array([-1, 1])[None, :]

    def __post_init__(self):
        pygame.init()

        h, w = (
            int(self.params.map_height_width[0]),
            int(self.params.map_height_width[1]),
        )

        self.CELL_SIZE = 40  # pixels per map unit

        self.bricks_texture = pygame.image.load("src/assets/brick.png")
        self.enemy_texture = pygame.image.load("src/assets/moving-enemy.png")
        self.bricks_texture = pygame.transform.scale(
            self.bricks_texture, (self.CELL_SIZE, self.CELL_SIZE)
        )
        self.enemy_texture = pygame.transform.scale(
            self.enemy_texture, (self.CELL_SIZE, self.CELL_SIZE)
        )

        self.screen = pygame.display.set_mode((w * self.CELL_SIZE, h * self.CELL_SIZE))
        pygame.display.set_caption("2D Env Renderer")

        # Correct collider radius in pixels
        self.COLLIDER_RADIUS_PX = self.CELL_SIZE // 2

        # Agent and goal match collider size
        self.AGENT_RADIUS_PX = self.COLLIDER_RADIUS_PX
        self.GOAL_RADIUS_PX = self.COLLIDER_RADIUS_PX

        # Obstacle square fits one grid cell
        self.OBSTACLE_SIDE_PX = self.CELL_SIZE

        self.clock = pygame.time.Clock()
        self.key = random.PRNGKey(0)
        self.obs, self.state = self.env.reset(self.key, self.params, self.init_state)
        self.running = True
        rotate_to_map_f = partial(
            utils.convert_to_map_view, map_shape=self.params.map_height_width
        )
        self.rotate_to_map_vmap = jax.vmap(rotate_to_map_f, in_axes=(0))

    def draw_square_with_striped_circle(
        self, center_px, square_side_px, circle_radius_px, square_color, num_stripes=6
    ):
        # Draw filled square (one cell)

        pygame.draw.rect(
            self.screen,
            square_color,
            (
                center_px[0] - square_side_px // 2,
                center_px[1] - square_side_px // 2,
                square_side_px,
                square_side_px,
            ),
        )
        # Draw stripes inside circle
        for i in range(-num_stripes, num_stripes + 1):
            y_offset = i * circle_radius_px / num_stripes
            half_width = math.sqrt(max(circle_radius_px**2 - y_offset**2, 0))
            start_pos = (int(center_px[0] - half_width), int(center_px[1] + y_offset))
            end_pos = (int(center_px[0] + half_width), int(center_px[1] + y_offset))
            pygame.draw.line(self.screen, BLACK, start_pos, end_pos, 1)

    def draw_path(self) -> None:
        """
        Draws a path over the map in blue
        """
        if self.path_array.shape[0] == 0:
            return  # nothing to draw

        waypoints = jnp.vstack([self.agent_pos, self.path_array])

        # Convert map positions to pixel coordinates
        pixels = [
            (int((wp[1] + 0.5) * self.CELL_SIZE), int((wp[0] + 0.5) * self.CELL_SIZE))
            for wp in waypoints
        ]

        # Draw lines connecting waypoints
        for start, end in zip(pixels[:-1], pixels[1:]):
            pygame.draw.line(self.screen, BLUE, start, end, 3)

    def draw_rays_transparent(self):
        """
        Draw transparent red rays from the agent's position.
        """
        # Ensure we have a valid surface
        screen_width, screen_height = self.screen.get_size()
        ray_surf = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)

        agent_px = (
            int((self.agent_pos[1] + 0.5) * self.CELL_SIZE),
            int((self.agent_pos[0] + 0.5) * self.CELL_SIZE),
        )

        rays = map(
            lambda x: (
                int(agent_px[0] + x[1] * self.CELL_SIZE),
                int(agent_px[1] + x[0] * self.CELL_SIZE),
            ),
            self.rays,
        )

        for ray in rays:
            pygame.draw.line(ray_surf, (255, 0, 0, self.alpha), agent_px, ray, 2)

        self.screen.blit(ray_surf, (0, 0))

    def draw_info(self):
        """
        Draws a dictionary of info on the screen.
        """
        font = pygame.font.SysFont("Arial", self.info_font_size)
        x, y = self.info_position

        for key, value in self.info.items():
            if key == "direction_of_path":
                value = f"(x={value[0]:.2f}, y={value[1]:.2f})"
            if key == "distance_to_path":
                value = f"{value:.4f}"
            text_surf = font.render(f"{key}: {value}", True, self.info_color)
            self.screen.blit(text_surf, (x, y))
            y += self.info_line_height

    def draw_kinematic_obstacles(self):
        for pos in self.kin_obst_map:
            y, x = pos
            center_px = (
                int(x * self.CELL_SIZE),
                int(y * self.CELL_SIZE),
            )
            self.screen.blit(self.enemy_texture, center_px)

    def draw_static_obstacles(self):
        for pos in self.static_obst_map:
            y, x = pos
            center_px = (
                int((x) * self.CELL_SIZE),
                int((y) * self.CELL_SIZE),
            )
            self.screen.blit(self.bricks_texture, center_px)

    def draw_goal(self):
        y, x = self.goal_pos
        center_px = (int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE))
        pygame.draw.circle(self.screen, GREEN, center_px, self.GOAL_RADIUS_PX)

    def draw_agent(self):
        # Agent itself
        y, x = self.agent_pos
        center_px = (int((x + 0.5) * self.CELL_SIZE), int((y + 0.5) * self.CELL_SIZE))
        pygame.draw.circle(self.screen, BLUE, center_px, self.AGENT_RADIUS_PX)

        # Agent's facing
        dir_vec = self.agent_forward_dir
        norm = math.sqrt(dir_vec[0] ** 2 + dir_vec[1] ** 2) + 1e-6
        dir_vec_px = (
            int(dir_vec[1] / norm * self.AGENT_RADIUS_PX),
            int(dir_vec[0] / norm * self.AGENT_RADIUS_PX),
        )
        end_pos = (center_px[0] + dir_vec_px[0], center_px[1] + dir_vec_px[1])
        pygame.draw.line(self.screen, YELLOW, center_px, end_pos, 3)

    def draw(self):
        self.screen.fill(WHITE)

        # self.draw_path()

        self.draw_kinematic_obstacles()
        self.draw_static_obstacles()

        self.draw_rays_transparent()

        self.draw_goal()
        self.draw_agent()

        self.draw_info()

        pygame.display.flip()

    def handle_keys(self):
        keys = pygame.key.get_pressed()
        action = jnp.array([0.0, 0.0], dtype=jnp.float32)
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            action += jnp.array([0, self.move_speed])
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            action += jnp.array([0, -self.move_speed])
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            action += jnp.array([-self.move_speed, 0])
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            action += jnp.array([self.move_speed, 0])

        action = (
            action / jnp.linalg.norm(action) * self.move_speed
            if jnp.any(jnp.abs(action) > 0)
            else action
        )

        return action

    def run(self):
        clock = pygame.time.Clock()

        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False

            if self.eval_mode and self.agent_fn is not None:
                action = self.agent_fn(self.state)

                self.dt = 1
                clock.tick(self.params.fps)

            else:
                action = self.handle_keys()

                self.dt = clock.tick(self.params.fps) / 1000

            self.obs, self.state, _, _, self.info = self.env.step(
                key=self.key,
                state=self.state,
                action=action,
                dt=self.dt,
                # self.params
            )

            self.draw()

        pygame.quit()
