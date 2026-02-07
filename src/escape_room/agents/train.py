from src.escape_room.gym_envs.escape_room_v1 import EscapeRoomEnv
import gymnasium as gym
from src.escape_room.agents.ppo import Trainer




def train_ppo(config:dict):
    gym.envs.register(
                id='EscapeRoomGame-v1',
                entry_point='__main__:EscapeRoomEnv',
            )

    env = gym.make(
            "EscapeRoomGame-v1",
            render_mode=config.get("render_mode", None),
            tmj_path="src/escape_room/assets/maps/level_two.tmj",
            tmx_path="src/escape_room/assets/maps/level_two.tmx",
            collision_layer_name="Collision",
            doors_layer_name="Doors",
            door_tiles_layer_name="DoorsTiles",
            rooms_layer_name="Rooms",   # <-- create this object layer in TMX for room reward
            time_limit_steps=config.get("time_limit_steps", 500),
        )
    config['state_dim'] = env.observation_space.shape[0]
    config['action_dim'] = env.action_space.n
    trainer = Trainer(env, config)
    trainer.train()