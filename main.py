from src.escape_room.envs.env import level_one, basic
from src.escape_room.gym_envs.escape_room_v1 import EscapeRoomEnv
from src.escape_room.agents.train import train_ppo
import gymnasium as gym



def main():
    gym.envs.register(
            id='EscapeRoomGame-v0',
            entry_point='__main__:EscapeRoomEnv',
        )

    env = gym.make(
            "EscapeRoomGame-v0",
            render_mode="human",
            tmj_path="src/escape_room/assets/maps/level_one.tmj",
            tmx_path="src/escape_room/assets/maps/level_one.tmx",
            collision_layer_name="Collision",
            doors_layer_name="Doors",
            door_tiles_layer_name="DoorsTiles",
            rooms_layer_name="Rooms",   # <-- create this object layer in TMX for room reward
            time_limit_steps=500,
        )
    obs, info = env.reset()
    print("Initial observation:", obs.shape)
    print("action space", env.action_space.n)
    done = False
    score = 0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # print(reward, info)
        score += reward
    print("Final score:", score)
    env.close()




def test_env():
    gym.envs.register(
        id="EscapeRoomGame-v0",
        entry_point="__main__:EscapeRoomEnv",
    )

    env = gym.make(
        "EscapeRoomGame-v0",
        render_mode="human",
        tmj_path="src/escape_room/assets/maps/level_one.tmj",
        tmx_path="src/escape_room/assets/maps/level_one.tmx",
        collision_layer_name="Collision",
        doors_layer_name="Doors",
        door_tiles_layer_name="DoorsTiles",
        rooms_layer_name="Rooms",
        time_limit_steps=500,
        # if you added these args in env:
        goal_layer_name="Goal_Room",
        goal_reward=10.0,
        terminate_on_goal=True,
    )

    obs, info = env.reset()
    print("Initial observation:", obs.shape)
    print("action space", env.action_space.n)

    # actions: 0 noop, 1 up, 2 down, 3 left, 4 right, 5 interact
    scripted_actions = (
        [1] * 60 +   # UP
        [3] * 50  +   # LEFT
        [5] * 5  +   # INTERACT
        [3] * 40      # LEFT
    )

    score = 0.0
    done = False

    for i, action in enumerate(scripted_actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += float(reward)

        print(f"step={i:03d} action={action} reward={reward:.2f} terminated={terminated} truncated={truncated} room={info.get('current_room')}")

        if done:
            print("Episode ended early (likely goal reached).")
            break

    if not done:
        print("Script finished but episode did NOT end. (Then goal rect / door / path not matching.)")

    print("Final score:", score)
    env.close()



def start_training():
    config = {
        "render_mode": None,
        "time_limit_steps": 500,
        'gamma': 0.99,
        'eps_clip': 0.2,
        'K_epochs': 4,
        'lr_actor': 1e-4,
        'lr_critic': 1e-3,
        'max_training_timesteps': int(3e6),  # Adjust as needed
        'max_ep_len': 1000,                  # Max timesteps per episode
        'update_timestep': 1000 * 4,            # How often to update PPO
        'log_freq': 1000 * 2,                   # How often to log
        'print_freq': 2000 * 5,                 # How often to print stats
        'save_model_freq': int(1e5)           # How often to save model
    }
    train_ppo(config)