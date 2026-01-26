# src/main.py
import sys
import pygame

from src.escape_room.core.tiles import TiledMap  # adjust import if your package name differs
from src.escape_room.core.player import Player, load_player_animations  # if player.py is also in src/
from src.escape_room.constants import *



def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def blit_fit(screen: pygame.Surface, world_surf: pygame.Surface):
    ww, wh = world_surf.get_size()
    sw, sh = screen.get_size()

    scale = min(sw / ww, sh / wh)  # fit fully inside screen
    new_w = int(ww * scale)
    new_h = int(wh * scale)

    scaled = pygame.transform.scale(world_surf, (new_w, new_h))
    x = (sw - new_w) // 2
    y = (sh - new_h) // 2

    screen.fill((0, 0, 0))
    screen.blit(scaled, (x, y))


def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("TMJ Fit-to-Screen Test")
    clock = pygame.time.Clock()

    # --- load map ---
    tmap = TiledMap("src/escape_room/assets/maps/finalized_escape.tmj")
    world_w = tmap.width * tmap.tile_w
    world_h = tmap.height * tmap.tile_h

    # --- pre-render map once (fast) ---
    map_surf = pygame.Surface((world_w, world_h), pygame.SRCALPHA)
    map_surf.fill((20, 20, 20))
    tmap.draw(map_surf, camera_x=0, camera_y=0)

    # --- load player ---
    anim = load_player_animations(
        idle_path=idle_path,
        walk_path=walk_path,
        run_path=run_path,
    )

    player = Player(
        pos=(world_w // 2, world_h // 2),  # world coords (feet position)
        animations=anim,
        scale=0.5,     # keep 1; whole world will be scaled to screen anyway
    )

    # working surface (world-sized) to compose frame: map + player
    world_frame = pygame.Surface((world_w, world_h), pygame.SRCALPHA)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0

        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

        keys = pygame.key.get_pressed()
        if keys[pygame.K_ESCAPE]:
            running = False

        # update player in world space
        player.update(dt, keys)

        # keep player inside world bounds (simple clamp)
        player.pos.x = max(0, min(world_w, player.pos.x))
        player.pos.y = max(0, min(world_h, player.pos.y))
        player._sync_rect_to_pos()  # update rect after clamp

        # compose: map + player into world_frame
        world_frame.blit(map_surf, (0, 0))
        world_frame.blit(player.image, player.rect.topleft)

        # fit whole world into 720x640
        blit_fit(screen, world_frame)
        pygame.display.flip()

    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
