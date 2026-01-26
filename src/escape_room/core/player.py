import pygame
from src.escape_room.utils.animations import SpriteSheet
from src.escape_room.constants import *




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
    """
    Modular player sprite:
    - update(dt, keys) handles movement + animation
    - draw(surface) draws with feet anchored on self.pos
    """

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

        self.walk_speed = float(walk_speed)
        self.run_speed = float(run_speed)

        self.pos = pygame.Vector2(pos)
        self.vel = pygame.Vector2(0, 0)

        self.direction = start_direction
        self.state = "idle"

        self.frame_index = 0
        self.timer = 0.0

        # init image/rect so Sprite groups can use it
        self.image = self._get_scaled_frame()
        self.rect = self.image.get_rect()
        self._sync_rect_to_pos()

    def handle_input(self, keys):
        """
        Returns:
            move_vec (Vector2), running (bool)
        """
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
        # 4-dir by dominant axis
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
        frame = self.anim[self.state][self.direction][self.frame_index]  # 48x96
        if self.scale == 1:
            return frame
        return pygame.transform.scale(frame, (FRAME_W * self.scale, FRAME_H * 2 * self.scale))

    def _sync_rect_to_pos(self):
        """
        Put 'feet' at self.pos:
        rect.midbottom = (pos.x, pos.y)
        """
        self.rect = self.image.get_rect()
        self.rect.midbottom = (int(self.pos.x), int(self.pos.y))

    def update(self, dt, keys):
        """
        dt: seconds (float)
        keys: pygame.key.get_pressed()
        """
        move, running = self.handle_input(keys)
        moving = move.length_squared() > 0

        if moving:
            self.direction = self._choose_direction(move)
            move = move.normalize()
            speed = self.run_speed if running else self.walk_speed
            self.pos += move * speed * dt

        self._set_state(moving, running)
        self._advance_animation(dt)

        self.image = self._get_scaled_frame()
        self._sync_rect_to_pos()

    def draw(self, surface):
        surface.blit(self.image, self.rect.topleft)
