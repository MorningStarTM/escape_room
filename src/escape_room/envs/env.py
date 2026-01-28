# src/main.py
import sys
import pygame
from src.escape_room.core.tiles import TiledMap
from src.escape_room.core.doors import build_doors_from_tiled
from src.escape_room.core.tiles import TiledMap, load_obstacle_rects_from_tmx  # adjust import if your package name differs
from src.escape_room.core.player import Player, load_player_animations  # if player.py is also in src/
from src.escape_room.constants import *
from src.escape_room.core.spatial_hash import SpatialHash
from src.escape_room.core.doors import Door



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




def nearest_door_to_interact(doors, player_rect, max_dist=50):
    best = None
    best_d2 = 10**18
    for d in doors:
        if d.can_interact(player_rect, max_dist_px=max_dist):
            dx = player_rect.centerx - d.trigger_rect.centerx
            dy = player_rect.centery - d.trigger_rect.centery
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best = d
    return best

def level_one():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Escape Room - Doors")
    clock = pygame.time.Clock()

    # --- load map ---
    tmap = TiledMap("src/escape_room/assets/maps/finalized_escape.tmj")
    world_w = tmap.width * tmap.tile_w
    world_h = tmap.height * tmap.tile_h

    # Walls/obstacles from TMX (your current method)
    wall_obstacles = load_obstacle_rects_from_tmx("src/escape_room/assets/maps/escape.tmx", "obstacle")

    # --- pre-render base map ONCE (everything except DoorsTiles) ---
    base_map = pygame.Surface((world_w, world_h), pygame.SRCALPHA)
    base_map.fill((20, 20, 20))

    # IMPORTANT:
    # your tmap.draw_cached draws everything. If DoorsTiles is included there, doors will never "disappear".
    # So you have two options:
    #  A) make sure DoorsTiles is NOT in the cached draw (best)
    #  B) don't use cached draw; draw each frame (slow)
    #
    # Here we assume your draw_cached does the base layers only.
    tmap.draw_cached(base_map, camera_x=0, camera_y=0)

    # --- doors from Tiled ---
    doors = build_doors_from_tiled(tmap, doors_layer_name="Doors", door_tiles_layer_name="DoorsTiles")

    # --- player ---
    anim = load_player_animations(
        idle_path=idle_path,
        walk_path=walk_path,
        run_path=run_path,
    )

    player = Player(
        pos=(world_w // 2, world_h // 2),
        animations=anim,
        scale=0.5,
    )

    # working surface (world size)
    world_frame = pygame.Surface((world_w, world_h), pygame.SRCALPHA)

    DEBUG_RAYS = True
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # ONLY ONCE

        # ----- events -----
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False

                elif e.key == pygame.K_r:
                    DEBUG_RAYS = not DEBUG_RAYS

                elif e.key == pygame.K_e:
                    door = nearest_door_to_interact(doors, player.rect, max_dist=60)
                    if door:
                        door.toggle()

        # ----- auto close doors (NOT inside event loop) -----
        for d in doors:
            d.update_auto_close(player.rect)

        # ----- build obstacles for THIS frame (walls + closed doors) -----
        door_blockers = []
        for d in doors:
            door_blockers.extend(d.blocker_rects())

        obstacles = wall_obstacles + door_blockers

        # rebuild spatial hash using current obstacles (simple + correct)
        grid = SpatialHash(cell_size=tmap.tile_w)
        grid.build(obstacles)

        # ----- update player (collides with obstacles INCLUDING doors) -----
        keys = pygame.key.get_pressed()
        player.update(dt, keys, obstacles, grid)

        # clamp world bounds
        player.pos.x = max(0, min(world_w, player.pos.x))
        player.pos.y = max(0, min(world_h, player.pos.y))
        player._sync_rect_to_pos()

        # ----- render -----
        world_frame.blit(base_map, (0, 0))

        # draw door tiles only if CLOSED:
        # we do this by drawing DoorsTiles layer but skipping the cells that belong to OPEN doors
        skip_cells = set()
        for d in doors:
            if d.open:
                skip_cells |= d.tiles_cells

        # draw the DoorsTiles layer on top (closed doors visible)
        tmap.draw_tile_layer(world_frame, "DoorsTiles", camera_x=0, camera_y=0, skip_cells=skip_cells)

        # player
        world_frame.blit(player.image, player.rect.topleft)

        if DEBUG_RAYS and hasattr(player, "ray_sensor") and player.ray_sensor:
            player.ray_sensor.draw(world_frame)

        blit_fit(screen, world_frame)
        pygame.display.flip()

    pygame.quit()
    sys.exit()




def basic():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Escape Room - Doors")
    clock = pygame.time.Clock()

    # --- load map ---
    tmap = TiledMap("src/escape_room/assets/maps/level_one.tmj")
    world_w = tmap.width * tmap.tile_w
    world_h = tmap.height * tmap.tile_h

    # Walls/obstacles from TMX (your current method)
    wall_obstacles = load_obstacle_rects_from_tmx("src/escape_room/assets/maps/level_one.tmx", "Collision")

    # --- pre-render base map ONCE (everything except DoorsTiles) ---
    base_map = pygame.Surface((world_w, world_h), pygame.SRCALPHA)
    base_map.fill((20, 20, 20))

    # IMPORTANT:
    # your tmap.draw_cached draws everything. If DoorsTiles is included there, doors will never "disappear".
    # So you have two options:
    #  A) make sure DoorsTiles is NOT in the cached draw (best)
    #  B) don't use cached draw; draw each frame (slow)
    #
    # Here we assume your draw_cached does the base layers only.
    tmap.draw_cached(base_map, camera_x=0, camera_y=0)

    # --- doors from Tiled ---
    doors = build_doors_from_tiled(tmap, doors_layer_name="Doors", door_tiles_layer_name="DoorsTiles")

    # --- player ---
    anim = load_player_animations(
        idle_path=idle_path,
        walk_path=walk_path,
        run_path=run_path,
    )

    player = Player(
        pos=(world_w // 2, world_h // 2),
        animations=anim,
        scale=0.5,
    )

    # working surface (world size)
    world_frame = pygame.Surface((world_w, world_h), pygame.SRCALPHA)

    DEBUG_RAYS = True
    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0  # ONLY ONCE

        # ----- events -----
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False

            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False

                elif e.key == pygame.K_r:
                    DEBUG_RAYS = not DEBUG_RAYS

                elif e.key == pygame.K_e:
                    door = nearest_door_to_interact(doors, player.rect, max_dist=60)
                    if door:
                        door.toggle()

        # ----- auto close doors (NOT inside event loop) -----
        for d in doors:
            d.update_auto_close(player.rect)

        # ----- build obstacles for THIS frame (walls + closed doors) -----
        door_blockers = []
        for d in doors:
            door_blockers.extend(d.blocker_rects())

        obstacles = wall_obstacles + door_blockers

        # rebuild spatial hash using current obstacles (simple + correct)
        grid = SpatialHash(cell_size=tmap.tile_w)
        grid.build(obstacles)

        # ----- update player (collides with obstacles INCLUDING doors) -----
        keys = pygame.key.get_pressed()
        player.update(dt, keys, obstacles, grid)

        # clamp world bounds
        player.pos.x = max(0, min(world_w, player.pos.x))
        player.pos.y = max(0, min(world_h, player.pos.y))
        player._sync_rect_to_pos()

        # ----- render -----
        world_frame.blit(base_map, (0, 0))

        # draw door tiles only if CLOSED:
        # we do this by drawing DoorsTiles layer but skipping the cells that belong to OPEN doors
        skip_cells = set()
        for d in doors:
            if d.open:
                skip_cells |= d.tiles_cells

        # draw the DoorsTiles layer on top (closed doors visible)
        tmap.draw_tile_layer(world_frame, "DoorsTiles", camera_x=0, camera_y=0, skip_cells=skip_cells)

        # player
        world_frame.blit(player.image, player.rect.topleft)

        if DEBUG_RAYS and hasattr(player, "ray_sensor") and player.ray_sensor:
            player.ray_sensor.draw(world_frame)

        blit_fit(screen, world_frame)
        pygame.display.flip()

    pygame.quit()
    sys.exit()
