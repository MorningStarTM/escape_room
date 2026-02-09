# src/escape_room/gym_envs/escape_room_env.py
from __future__ import annotations

import os
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List, Set
import math
import numpy as np
import pygame
import gymnasium as gym
from gymnasium import spaces

from src.escape_room.core.tiles import TiledMap, load_obstacle_rects_from_tmx
from src.escape_room.core.player import Player, load_player_animations
from src.escape_room.core.spatial_hash import SpatialHash
from src.escape_room.core.doors import build_doors_from_tmx
from src.escape_room.core.rays import RaySensor180

from src.escape_room.constants import (
    SCREEN_W, SCREEN_H, FPS,
    idle_path, walk_path, run_path,
)


# -------------------- helpers copied from your main --------------------

def blit_fit(screen: pygame.Surface, world_surf: pygame.Surface):
    ww, wh = world_surf.get_size()
    sw, sh = screen.get_size()

    scale = min(sw / ww, sh / wh)
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




@dataclass
class Lever:
    lever_id: str
    rect: pygame.Rect
    on: bool = False

    def can_interact(self, player_rect: pygame.Rect, max_dist_px: int = 60) -> bool:
        dx = player_rect.centerx - self.rect.centerx
        dy = player_rect.centery - self.rect.centery
        return (dx * dx + dy * dy) <= (max_dist_px * max_dist_px)

    def toggle(self):
        self.on = not self.on


def load_levers_from_tmx(tmx_path: str, layer_name: str = "Liver") -> List[Lever]:
    """
    Reads TMX objectgroup named 'Liver' (your screenshot) and returns Lever rects.
    Also supports fallback to 'Lever' if 'Liver' not found.
    """
    tree = ET.parse(tmx_path)
    root = tree.getroot()

    def _read_layer(name: str) -> List[Lever]:
        levers: List[Lever] = []
        for og in root.findall("objectgroup"):
            if og.get("name") != name:
                continue
            for idx, obj in enumerate(og.findall("object")):
                x = int(float(obj.get("x", "0")))
                y = int(float(obj.get("y", "0")))
                w = int(float(obj.get("width", "0")))
                h = int(float(obj.get("height", "0")))
                if w <= 0 or h <= 0:
                    continue
                oid = obj.get("name") or f"{name}_{idx}"
                levers.append(Lever(lever_id=str(oid), rect=pygame.Rect(x, y, w, h), on=False))
            break
        return levers

    levers = _read_layer(layer_name)
    if not levers and layer_name == "Liver":
        levers = _read_layer("Lever")  # fallback
    return levers


def nearest_lever_to_interact(levers: List[Lever], player_rect: pygame.Rect, max_dist: int = 60) -> Lever | None:
    best = None
    best_d2 = 10**18
    for lv in levers:
        if lv.can_interact(player_rect, max_dist_px=max_dist):
            dx = player_rect.centerx - lv.rect.centerx
            dy = player_rect.centery - lv.rect.centery
            d2 = dx*dx + dy*dy
            if d2 < best_d2:
                best_d2 = d2
                best = lv
    return best



# -------------------- room loader (TMX objectgroup) --------------------

@dataclass(frozen=True)
class Room:
    room_id: str
    rect: pygame.Rect


def load_rooms_from_tmx(tmx_path: str, rooms_layer_name: str = "Rooms") -> List[Room]:
    rooms: List[Room] = []
    tree = ET.parse(tmx_path)
    root = tree.getroot()

    for og in root.findall("objectgroup"):
        if og.get("name") != rooms_layer_name:
            continue
        for idx, obj in enumerate(og.findall("object")):
            x = int(float(obj.get("x", "0")))
            y = int(float(obj.get("y", "0")))
            w = int(float(obj.get("width", "0")))
            h = int(float(obj.get("height", "0")))
            if w <= 0 or h <= 0:
                continue
            rid = obj.get("name") or f"Room_{idx}"
            rooms.append(Room(room_id=str(rid), rect=pygame.Rect(x, y, w, h)))
        break

    return rooms




# -------------------- Key proxy (so we DON'T change your Player.update) --------------------

class KeyProxy:
    """
    Minimal object that behaves like pygame.key.get_pressed() for Player.update().
    Player.update likely does: keys[pygame.K_w] etc.
    """
    def __init__(self, pressed_keys: Set[int]):
        self._pressed = pressed_keys

    def __getitem__(self, key: int) -> bool:
        return key in self._pressed


# -------------------- The Gym Env --------------------

class EscapeRoomEnv(gym.Env):
    """
    Gymnasium wrapper around your real Escape Room game (TMJ + TMX + Player + Doors).

    Reward (as you said):
      - time exceed -> terminated True, reward -10
      - enter room first time -> +5
      - enter room again -> -0.5
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": FPS}

    def __init__(
        self,
        render_mode: Optional[str] = None,
        tmj_path: str = "src/escape_room/assets/maps/level_two.tmj",
        tmx_path: str = "src/escape_room/assets/maps/level_two.tmx",
        collision_layer_name: str = "Collision",
        doors_layer_name: str = "Doors",
        door_tiles_layer_name: str = "DoorsTiles",
        rooms_layer_name: str = "Rooms",
        goal_layer_name: str = "Goal_Room",      # NEW (matches your layer name)
        goal_reward: float = 10.0,               # NEW
        terminate_on_goal: bool = True,
        time_limit_steps: int = 1000,
        max_interact_dist: int = 60,
        player_spawn: Optional[Tuple[int, int]] = None,
        debug_rays: bool = True,
        
    ):
        super().__init__()
        assert render_mode in (None, "human", "rgb_array")
        self.render_mode = render_mode

        self.tmj_path = tmj_path
        self.tmx_path = tmx_path

        self.collision_layer_name = collision_layer_name
        self.doors_layer_name = doors_layer_name
        self.door_tiles_layer_name = door_tiles_layer_name
        self.rooms_layer_name = rooms_layer_name

        self.goal_layer_name = goal_layer_name
        self.goal_reward = float(goal_reward)
        self.terminate_on_goal = bool(terminate_on_goal)

        # NEW goal state
        self.goal_rects: List[pygame.Rect] = []
        self._goal_reached = False

        self.time_limit_steps = int(time_limit_steps)
        self.max_interact_dist = int(max_interact_dist)
        self.debug_rays = bool(debug_rays)

        # actions: 0 noop, 1 up, 2 down, 3 left, 4 right, 5 interact
        self.action_space = spaces.Discrete(7)

        # levers + goal unlock rule
        self.levers: List[Lever] = []
        self.required_levers_on: int = 0

        # observation: you can replace later with real rays etc.
        # For now: [px_norm, py_norm] only (very simple but valid)
        self.n_rays = 31                 # choose 31 / 61 / 91 etc.
        self.ray_max_dist = 300.0        # pixels, must match your world scale
        self.ray_sensor = RaySensor180(
            fov_deg=180.0,
            n_rays=self.n_rays,
            max_dist=self.ray_max_dist,
        )

        # observation: normalized ray distances in [0,1]
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(33,), dtype=np.float32)


        # pygame render
        self._screen: Optional[pygame.Surface] = None
        self._clock: Optional[pygame.time.Clock] = None

        # world objects initialized in reset()
        self.tmap: Optional[TiledMap] = None
        self.world_w = 0
        self.world_h = 0

        self.base_map: Optional[pygame.Surface] = None
        self.world_frame: Optional[pygame.Surface] = None

        self.wall_obstacles: List[pygame.Rect] = []
        self.doors = []
        self.rooms: List[Room] = []

        self.player: Optional[Player] = None
        self._spawn = player_spawn  # if None, spawn center

        # reward tracking
        self.step_count = 0
        self.visited_rooms: Set[str] = set()
        self._current_room_id: Optional[str] = None  # for entry-based reward

        self._pending_room_id = None
        self._pending_room_steps = 0
        self._room_confirm_steps = 6   

    # ---------------- Gym API ----------------

    def _player_facing_angle_rad(self) -> float:
        # Player.direction is one of: "up", "down", "left", "right"
        d = self.player.direction
        if d == "right":
            return 0.0
        if d == "down":
            return math.pi / 2.0
        if d == "left":
            return math.pi
        return -math.pi / 2.0  # "up"



    def _get_ray_obstacles(self):
        obstacles = list(self.wall_obstacles)

        # If you have door rects, include only the closed ones
        # (adjust attribute names depending on your Door class)
        for door in self.doors:
            if door.open:
                continue
            obstacles.append(door.blocker_rect)

        return obstacles



    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        super().reset(seed=seed)

        if not pygame.get_init():
            pygame.init()

        if not pygame.display.get_init():
            pygame.display.init()

        # If no display surface exists yet, create a tiny hidden one.
        # This is required so pygame.Surface.convert_alpha() works.
        if pygame.display.get_surface() is None:
            pygame.display.set_mode((1, 1), flags=pygame.HIDDEN)
        # init pygame for rendering if needed
        if self.render_mode is not None and not pygame.get_init():
            pygame.init()

        # load map
        self.tmap = TiledMap(self.tmj_path)
        self.world_w = self.tmap.width * self.tmap.tile_w
        self.world_h = self.tmap.height * self.tmap.tile_h

        # obstacles from TMX
        self.wall_obstacles = load_obstacle_rects_from_tmx(self.tmx_path, self.collision_layer_name)

        # doors from TMX (your earlier fix)
        # pass tmap so door tiles cells can be computed if DoorsTiles exists in TMJ
        self.doors = build_doors_from_tmx(
            self.tmx_path,
            tmap=self.tmap,
            doors_layer_name=self.doors_layer_name,
            door_tiles_layer_name=self.door_tiles_layer_name,
        )
        # print("[DOORS DEBUG] count =", len(self.doors))
        # print("[DOORS DEBUG] goal doors =", [(getattr(d, "is_goal", False), d.blocker_rect) for d in self.doors])


        #print("[DOORS] count =", len(self.doors))
        # rooms from TMX (for your reward)
        self.rooms = load_rooms_from_tmx(self.tmx_path, rooms_layer_name=self.rooms_layer_name)
        #print("[ROOMS] count =", len(self.rooms), "sample ids =", [r.room_id for r in self.rooms[:10]])

        self.goal_rects = self._load_object_rects_from_tmx(self.tmx_path, self.goal_layer_name)
        self._goal_reached = False
        gr = self.goal_rects[0]
        self.goal_center = pygame.Vector2(gr.centerx, gr.centery)
        #print("[GOAL] rects:", [(r.x, r.y, r.w, r.h) for r in self.goal_rects])

        # levers from TMX (layer name in your screenshot is "Liver")
        self.levers = load_levers_from_tmx(self.tmx_path, layer_name="Liver")

        # rule: required = no_room - 3   (no_room is always odd, as you said)
        no_rooms = len(self.rooms)
        self.required_levers_on = max(0, no_rooms - 3)

        # reset lever states explicitly
        for lv in self.levers:
            lv.on = False

        

        # pre-render base map once
        self.base_map = pygame.Surface((self.world_w, self.world_h), pygame.SRCALPHA)
        #self.base_map.fill((20, 20, 20))
        #self.tmap.draw_cached(self.base_map, camera_x=0, camera_y=0)
        # Build base_map WITHOUT DoorsTiles.
        # We try to draw each tile layer except DoorsTiles.
        # (This depends on your TiledMap internals, so we log and fallback.)
        self.base_map.fill((20, 20, 20))

        try:
            # If your TiledMap exposes layer names like tmap.layers or tmap.layer_names:
            layer_names = []
            if hasattr(self.tmap, "layers") and isinstance(self.tmap.layers, dict):
                layer_names = list(self.tmap.layers.keys())
            elif hasattr(self.tmap, "layer_names"):
                layer_names = list(self.tmap.layer_names)

            #print(f"[MAP] layer_names={layer_names}")

            # draw all layers except DoorsTiles
            for lname in layer_names:
                if lname == self.door_tiles_layer_name:
                    continue
                self.tmap.draw_tile_layer(self.base_map, lname, camera_x=0, camera_y=0)

            #print(f"[MAP] base_map rendered without '{self.door_tiles_layer_name}'")
        except Exception as e:
            #print(f"[MAP] Could not build base_map layer-by-layer: {e}")
            #print("[MAP] FALLBACK: base_map will include DoorsTiles, door visuals may not disappear.")
            self.tmap.draw_cached(self.base_map, camera_x=0, camera_y=0)



        # world working surface
        self.world_frame = pygame.Surface((self.world_w, self.world_h), pygame.SRCALPHA)
        for gr in self.goal_rects:
            pygame.draw.rect(self.world_frame, (255, 0, 0), gr, 2)
        # player
        anim = load_player_animations(idle_path=idle_path, walk_path=walk_path, run_path=run_path)

        spawn = self._spawn
        if spawn is None:
            spawn = (self.world_w // 2, self.world_h // 2)

        self.player = Player(pos=spawn, animations=anim, scale=0.5)

        # reset trackers
        self.step_count = 0
        self.visited_rooms.clear()
        self._current_room_id = self._room_id_for_player(self.player.rect)  # may be None
        if self._current_room_id is not None:
            # starting room counts as "visited" but no reward on reset
            self.visited_rooms.add(self._current_room_id)

        obs = self._get_obs()
        info = {}

        # render first frame so window isn't blank
        if self.render_mode == "human":
            self.render()

        return obs, info

    def step(self, action: int):
        assert self.player is not None
        assert self.tmap is not None
        assert self.world_frame is not None

        self.step_count += 1

        # ----- time exceeded => end with -10 -----
        terminated = False
        truncated = False
        reward = 0.0

        if self.step_count >= self.time_limit_steps:
            terminated = True
            reward = -10.0
            obs = self._get_obs()
            info = self._get_info()
            if self.render_mode == "human":
                self.render()
            return obs, reward, terminated, truncated, info

        # ----- apply action -> keys for your existing Player.update() -----
        pressed = set()

        if action == 1:
            pressed.add(pygame.K_w); pressed.add(pygame.K_UP)
        elif action == 2:
            pressed.add(pygame.K_s); pressed.add(pygame.K_DOWN)
        elif action == 3:
            pressed.add(pygame.K_a); pressed.add(pygame.K_LEFT)
        elif action == 4:
            pressed.add(pygame.K_d); pressed.add(pygame.K_RIGHT)

        elif action == 5:
            door = nearest_door_to_interact(self.doors, self.player.rect, max_dist=self.max_interact_dist)
            if door is not None:
                on_count = sum(1 for lv in self.levers if lv.on)
                goal_unlocked = (on_count >= self.required_levers_on)

                # if this is the goal door, don’t allow toggle until unlocked
                if getattr(door, "is_goal", False) and not goal_unlocked:
                    pass
                else:
                    door.toggle()

        elif action == 6:
            # LEVER TOGGLE (NEW)
            lv = nearest_lever_to_interact(
                self.levers, self.player.rect, max_dist=self.max_interact_dist
            )
            if lv is not None:
                lv.toggle()
                #print(f"[LEVER] toggled {lv.lever_id} to {'ON' if lv.on else 'OFF'}")


        keys = KeyProxy(pressed)

        # ----- door auto close -----
        for d in self.doors:
            d.update_auto_close(self.player.rect)

        # ----- obstacles this frame (walls + closed doors) -----
        door_blockers: List[pygame.Rect] = []
        for d in self.doors:
            door_blockers.extend(d.blocker_rects())

        obstacles = self.wall_obstacles + door_blockers

        # spatial hash
        grid = SpatialHash(cell_size=self.tmap.tile_w)
        grid.build(obstacles)

        # ----- update player with collisions (UNCHANGED player.update) -----
        dt = 1.0 / float(FPS)
        self.player.update(dt, keys, obstacles, grid)

        # clamp world bounds
        self.player.pos.x = max(0, min(self.world_w, self.player.pos.x))
        self.player.pos.y = max(0, min(self.world_h, self.player.pos.y))
        self.player._sync_rect_to_pos()

        # ----- reward: room visit (ENTRY-BASED) -----
        room_r = self._room_entry_reward()
        reward += room_r

        # unlock status
        on_count = sum(1 for lv in self.levers if lv.on)
        goal_unlocked = (on_count >= self.required_levers_on)

        # goal reward ONLY if unlocked
        goal_hit = any(gr.colliderect(self.player.rect) for gr in self.goal_rects)
        goal_r = 0.0

        if goal_unlocked and goal_hit and not self._goal_reached:
            goal_r = self.goal_reward
            reward += goal_r
            self._goal_reached = True
            if self.terminate_on_goal:
                terminated = True

        obs = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()


        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None

        if self._screen is None:
            if not pygame.get_init():
                pygame.init()
            os.environ["SDL_VIDEO_CENTERED"] = "1"
            self._screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
            pygame.display.set_caption("Escape Room (Gym Env)")
            self._clock = pygame.time.Clock()

        # keep window responsive
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                self.close()
                return None

        assert self.world_frame is not None
        assert self.tmap is not None
        assert self.player is not None

        # ---------- draw world_frame ----------
        self.world_frame.fill((20, 20, 20))
        self.tmap.draw_cached(self.world_frame, camera_x=0, camera_y=0)

        # draw DoorsTiles if CLOSED (skip open)
        skip_cells = set()
        for d in self.doors:
            if d.open:
                skip_cells |= d.tiles_cells

        try:
            self.tmap.draw_tile_layer(
                self.world_frame,
                self.door_tiles_layer_name,
                camera_x=0,
                camera_y=0,
                skip_cells=skip_cells,
            )
        except Exception:
            pass

        # player
        self.world_frame.blit(self.player.image, self.player.rect.topleft)

        # your ray sensor (ONLY ONCE)
        if self.debug_rays:
            # draw the env ray sensor you actually update in _get_obs()
            self.ray_sensor.draw(self.world_frame)

        # levers
        for lv in self.levers:
            color = (0, 200, 0) if lv.on else (200, 0, 0)
            pygame.draw.rect(self.world_frame, color, lv.rect, 2)

        # doors debug
        for d in self.doors:
            col = (255, 255, 0) if getattr(d, "is_goal", False) else (0, 150, 255)
            if d.open:
                col = (100, 100, 100)
            pygame.draw.rect(self.world_frame, col, d.blocker_rect, 2)
            pygame.draw.rect(self.world_frame, (255, 0, 255), d.trigger_rect, 1)

        # ---------- present to screen ----------
        blit_fit(self._screen, self.world_frame)
        pygame.display.flip()

        if self._clock is not None:
            self._clock.tick(self.metadata["render_fps"])

        # ---------- return rgb_array AFTER presenting ----------
        if self.render_mode == "rgb_array":
            arr = pygame.surfarray.array3d(self._screen)  # (W,H,3)
            return np.transpose(arr, (1, 0, 2))  # (H,W,3)

        return None


    def close(self):
        if self._screen is not None:
            pygame.quit()
        self._screen = None
        self._clock = None

    # ---------------- internal helpers ----------------

    def _get_obs(self):
        assert self.player is not None

        # 1) update ray sensor each step/reset
        origin = pygame.Vector2(self.player.rect.centerx, self.player.rect.centery)
        facing = self._player_facing_angle_rad()
        obstacles = self._get_ray_obstacles()

        self.ray_sensor.update(origin, facing, obstacles)

        # 2) distances -> numpy
        rays = np.array(self.ray_sensor.distances, dtype=np.float32)  # (31,)
        rays = rays / float(self.ray_sensor.max_dist)                 # normalize to [0,1]

        # 3) goal relative position
        px, py = self.player.rect.centerx, self.player.rect.centery
        dx = (self.goal_center.x - px) / float(self.world_w)
        dy = (self.goal_center.y - py) / float(self.world_h)

        obs = np.concatenate([rays, np.array([dx, dy], dtype=np.float32)], axis=0)
        return obs


    def _get_info(self) -> Dict[str, Any]:
        on_count = sum(1 for lv in self.levers if lv.on)
        return {
            "step": self.step_count,
            "visited_rooms": len(self.visited_rooms),
            "current_room": self._current_room_id,
            "lever_on": on_count,
            "lever_required": self.required_levers_on,
            "goal_unlocked": (on_count >= self.required_levers_on),
        }


    def _room_id_for_player(self, player_rect: pygame.Rect) -> Optional[str]:
        # Use feet point (stable for top-down)
        p = player_rect.midbottom
        for r in self.rooms:
            if r.rect.collidepoint(player_rect.midbottom):
                return r.room_id
        return None

    def _room_entry_reward(self) -> float:
        now = self._room_id_for_player(self.player.rect)

        # SAME state (including None)
        if now == self._current_room_id:
            self._pending_room_id = None
            self._pending_room_steps = 0
            return 0.0

        # start / continue debounce (for both room and None)
        if now != self._pending_room_id:
            self._pending_room_id = now
            self._pending_room_steps = 1
            return 0.0

        self._pending_room_steps += 1
        if self._pending_room_steps < self._room_confirm_steps:
            return 0.0

        # ---- CONFIRMED TRANSITION ----
        prev = self._current_room_id
        self._current_room_id = now
        self._pending_room_id = None
        self._pending_room_steps = 0

        # exiting to None → no reward, but state updates!
        if now is None:
            return 0.0

        # first time entering this room
        if now not in self.visited_rooms:
            self.visited_rooms.add(now)
            return 1.0

        # re-entering visited room → penalty
        return -0.5

        
    
    def _load_object_rects_from_tmx(self, tmx_path: str, layer_name: str) -> List[pygame.Rect]:
        """
        Reads <objectgroup name="layer_name"> from TMX and returns pygame.Rect list.
        Supports rectangle objects (x,y,width,height).
        """
        rects: List[pygame.Rect] = []

        tree = ET.parse(tmx_path)
        root = tree.getroot()

        # object layers are <objectgroup>
        for og in root.findall("objectgroup"):
            if og.attrib.get("name") != layer_name:
                continue

            for obj in og.findall("object"):
                x = float(obj.attrib.get("x", "0"))
                y = float(obj.attrib.get("y", "0"))
                w = float(obj.attrib.get("width", "0"))
                h = float(obj.attrib.get("height", "0"))

                # skip points/empty objects
                if w <= 0 or h <= 0:
                    continue

                rects.append(pygame.Rect(int(x), int(y), int(w), int(h)))

        return rects
