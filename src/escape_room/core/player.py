import math
import pygame
from src.escape_room.utils.animations import SpriteSheet
from src.escape_room.constants import *
from src.escape_room.core.rays import RaySensor180



def load_player_animations(
    idle_path="48Free/Male/Male_I.png",
    walk_path="48Free/Male/Male_W.png",
    run_path="48Free/Male/Male_R.png",
    start_col=0,
    end_col=8,
):
    """
    Returns:
        anim[state][direction] = list[Surface] (each is 48x96)
    """
    idle = SpriteSheet(idle_path)
    walk = SpriteSheet(walk_path)
    run  = SpriteSheet(run_path)

    anim = {"idle": {}, "walk": {}, "run": {}}

    for d, top_row in DIR_TOPROW.items():
        anim["idle"][d] = idle.get_full_row_frames(top_row, start_col, end_col)
        anim["walk"][d] = walk.get_full_row_frames(top_row, start_col, end_col)
        anim["run"][d]  = run.get_full_row_frames(top_row, start_col, end_col)

    return anim



class Player(pygame.sprite.Sprite):
    def __init__(
        self,
        pos,
        animations,
        scale=1,
        walk_speed=140,
        run_speed=220,
        start_direction="down",
    ):
        super().__init__()

        self.anim = animations
        self.scale = scale
        self.anim_scaled = self._build_scaled_anim_cache()


        self.walk_speed = float(walk_speed)
        self.run_speed = float(run_speed)

        self.pos = pygame.Vector2(pos)  # this is the FEET position (midbottom)
        self.vel = pygame.Vector2(0, 0)

        self.direction = start_direction
        self.state = "idle"

        self.frame_index = 0
        self.timer = 0.0

        self.image = self._get_scaled_frame()
        self.rect = self.image.get_rect()
        self._sync_rect_to_pos()

        # --- collision hitbox (smaller than sprite) ---
        self.hitbox = pygame.Rect(0, 0, 20 * self.scale, 14 * self.scale)
        self._sync_hitbox_to_pos()

        self._ray_timer = 0.0
        self._ray_period = 1.0 / 15.0   # 15 Hz ray update (smooth enough)
        self._last_ray_pos = pygame.Vector2(self.pos)
        self._last_ray_dir = self.direction

        self.ray_sensor = RaySensor180(
                n_rays=41,
                max_dist=220,
                fov_deg=180
            )



    def _build_scaled_anim_cache(self):
        """Pre-scale all frames once. Huge speed-up."""
        if self.scale == 1:
            return self.anim  # no need

        scaled = {"idle": {}, "walk": {}, "run": {}}
        for state in self.anim:
            for d in self.anim[state]:
                scaled[state][d] = []
                for frame in self.anim[state][d]:
                    scaled[state][d].append(
                        pygame.transform.scale(
                            frame,
                            (FRAME_W * self.scale, FRAME_H * 2 * self.scale)
                        )
                    )
        return scaled



    def handle_input(self, keys):
        move = pygame.Vector2(0, 0)
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            move.x -= 1
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            move.x += 1
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            move.y -= 1
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            move.y += 1
        running = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
        return move, running

    def _choose_direction(self, move):
        if abs(move.x) > abs(move.y):
            return "right" if move.x > 0 else "left"
        else:
            return "down" if move.y > 0 else "up"

    def _set_state(self, moving, running):
        if not moving:
            self.state = "idle"
        else:
            self.state = "run" if running else "walk"

    def _advance_animation(self, dt):
        fps = ANIM_FPS[self.state]
        frames = self.anim[self.state][self.direction]
        self.timer += dt
        step = 1.0 / float(fps)
        while self.timer >= step:
            self.timer -= step
            self.frame_index = (self.frame_index + 1) % len(frames)

    def _get_scaled_frame(self):
        return self.anim_scaled[self.state][self.direction][self.frame_index]


    def _sync_rect_to_pos(self):
        self.rect.midbottom = (int(self.pos.x), int(self.pos.y))


    def _sync_hitbox_to_pos(self):
        # hitbox also uses "feet anchor"
        self.hitbox.midbottom = (int(self.pos.x), int(self.pos.y))

    def _resolve_collisions(self, obstacles, dx, dy):
        """
        Push hitbox out of obstacles and update self.pos accordingly.
        """
        if not obstacles:
            return

        # X axis resolution
        if dx != 0:
            for r in obstacles:
                if self.hitbox.colliderect(r):
                    if dx > 0:
                        self.hitbox.right = r.left
                    else:
                        self.hitbox.left = r.right
            self.pos.x = self.hitbox.centerx  # keep feet x aligned

        # Y axis resolution
        if dy != 0:
            for r in obstacles:
                if self.hitbox.colliderect(r):
                    if dy > 0:
                        self.hitbox.bottom = r.top
                    else:
                        self.hitbox.top = r.bottom
            self.pos.y = self.hitbox.bottom  # feet = bottom of hitbox



    def _direction_to_angle_rad(self, direction: str) -> float:
        # 0 rad = facing right (pygame +x). Positive is clockwise? (we’ll follow standard)
        if direction == "right":
            return 0.0
        if direction == "down":
            return math.pi / 2
        if direction == "left":
            return math.pi
        if direction == "up":
            return -math.pi / 2
        return 0.0



    def update(self, dt, keys, obstacles=None, grid=None):
        """
        dt: seconds
        obstacles: list[pygame.Rect] from TMX object layer "obstacle"
        """
        if obstacles is None:
            obstacles = []

        move, running = self.handle_input(keys)
        moving = move.length_squared() > 0

        if moving:
            self.direction = self._choose_direction(move)
            move = move.normalize()
            speed = self.run_speed if running else self.walk_speed

            dx = move.x * speed * dt
            dy = move.y * speed * dt

            # --- move + collide (separate axes) ---
            self.pos.x += dx
            self._sync_hitbox_to_pos()
            self._resolve_collisions(obstacles, dx=dx, dy=0)

            self.pos.y += dy
            self._sync_hitbox_to_pos()
            self._resolve_collisions(obstacles, dx=0, dy=dy)

        self._set_state(moving, running)
        self._advance_animation(dt)

        self.image = self._get_scaled_frame()
        self._sync_rect_to_pos()
        self._sync_hitbox_to_pos()

        self._ray_timer += dt

        moved  = (self.pos - self._last_ray_pos).length_squared() > 1.0
        turned = (self.direction != self._last_ray_dir)

        if (moved or turned) and (self._ray_timer >= self._ray_period):
            self._ray_timer = 0.0
            self._last_ray_pos.update(self.pos)
            self._last_ray_dir = self.direction

            # ✅ ONLY update rays here
            facing_rad = self._direction_to_angle_rad(self.direction)
            if grid is not None:
                max_dist = self.ray_sensor.max_dist
                sense_rect = pygame.Rect(
                    int(self.pos.x - max_dist), int(self.pos.y - max_dist),
                    int(max_dist * 2), int(max_dist * 2)
                )
                nearby = grid.query_rect(sense_rect)
            else:
                nearby = obstacles

            self.ray_sensor.update(self.pos, facing_rad, nearby)




    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)
        # debug hitbox (optional)
        # pygame.draw.rect(surface, (255,0,0), self.hitbox, 1)
