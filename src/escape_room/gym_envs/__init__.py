# src/escape_room/gym_envs/__init__.py
from gymnasium.envs.registration import register
from src.escape_room.gym_envs.escape_room_v1 import EscapeRoomEnv

register(
    id="EscapeRoomGame-v1",
    entry_point="__main__:EscapeRoomEnv",
)
