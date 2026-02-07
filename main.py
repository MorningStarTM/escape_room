import argparse
import gymnasium as gym

from src.escape_room.gym_envs.escape_room_v1 import EscapeRoomEnv
from src.escape_room.agents.train import train_ppo
from src.escape_room.agents.ppo import PPO
from src.escape_room.utils.eval import evaluate_agent_and_save_gifs


def make_env(render_mode="human"):
    # register once (safe even if called multiple times in same process)
    try:
        gym.envs.register(
            id="EscapeRoomGame-v0",
            entry_point="__main__:EscapeRoomEnv",
        )
    except Exception:
        pass  # already registered

    env = gym.make(
        "EscapeRoomGame-v0",
        render_mode=render_mode,
        tmj_path="src/escape_room/assets/maps/level_two.tmj",
        tmx_path="src/escape_room/assets/maps/level_two.tmx",
        collision_layer_name="Collision",
        doors_layer_name="Doors",
        door_tiles_layer_name="DoorsTiles",
        rooms_layer_name="Rooms",
        time_limit_steps=3000,
        # optional goal args (only if your env supports them)
        goal_layer_name="Goal_Room",
        goal_reward=10.0,
        terminate_on_goal=True,
    )
    return env


def test_random(render_mode="human"):
    env = make_env(render_mode=render_mode)
    obs, info = env.reset()
    print("Initial observation:", obs.shape)
    print("action space", env.action_space.n)

    done = False
    score = 0.0
    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += float(reward)

    print("Final score:", score)
    env.close()


def test_scripted(render_mode="human"):
    env = make_env(render_mode=render_mode)
    obs, info = env.reset()
    print("Initial observation:", obs.shape)
    print("action space", env.action_space.n)

    # actions: 0 noop, 1 up, 2 down, 3 left, 4 right, 5 interact
    scripted_actions = (
        [1] * 12 +   # UP
        [3] * 50 +   # LEFT
        [5] * 15 +   # INTERACT
        [3] * 40 +    # LEFT
        [0] * 50 +
        [4] * 60 +   # right
        [5] * 15 +    # INTERACT
        [4] * 160 +    # RIGHT
        [0] * 50 +
        [3] * 80 +   # LEFT
        [5] * 15 +   # INTERACT
        [3] * 40     # LEFT

        
    )

    score = 0.0
    done = False
    for i, action in enumerate(scripted_actions, start=1):
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += float(reward)

        # print(
        #     f"step={i:03d} action={action} reward={reward:.2f} "
        #     f"terminated={terminated} truncated={truncated} room={info.get('current_room')}"
        # )

        if done:
            print("Episode ended early (likely goal reached).")
            break

    if not done:
        print("Script finished but episode did NOT end. (Goal rect / door / path mismatch.)")

    print("Final score:", score)
    env.close()


def test_scripted_hardcoded(render_mode="human"):
    env = make_env(render_mode=render_mode)
    obs, info = env.reset()

    scripted_actions = (
        # go to room 1
        
        #[2]*70 + [3]*20 + [2]*15 + [5]*15 + [2]*20 + [5]*15 + [2]*20 
         [1]*12 + [4]*65 + [5]*15 + [4]*35 + [2]*10 + [6]*1 + [1]*10 + [3]*12 + [5]*15 + [3]*40 +
         [1]*40 + [4]*65 + [5]*15 + [4]*50 + [6]*1 + [3]*20 + [5]*15 + [3]*42 +  
         [2]*155 + [5]*15 + [2]*30 + [4]*10 + [6]*1 + [3]*10 + [1]*30 +
         [3]*29 + [2]*35 + [5]*15 + [2]*20 + [3]*20 + [6]*1 + [4]*24 + [1]*20 + [5]*15 + [1]*30 +
         [1]*60 + [3]*20 + [5]*15 + [3]*40 
        #[3]*20 + [2]*80 + [5]*15 + [2]*10
    )

    score = 0.0
    for i, action in enumerate(scripted_actions):
        obs, reward, terminated, truncated, info = env.step(action)
        score += reward

        if reward != 0:
            print(
                f"step={i} reward={reward:.2f} "
                f"levers_on={info['lever_on']} unlocked={info['goal_unlocked']}"
            )

        if terminated or truncated:
            print("Goal reached / episode ended")
            break

    print("Final score:", score)
    env.close()



def agent_training(render_mode=None):
    config = {
        "render_mode": render_mode,     # None for faster training, "human" to watch
        "time_limit_steps": 500,
        "gamma": 0.99,
        "eps_clip": 0.2,
        "K_epochs": 4,
        "lr_actor": 1e-4,
        "lr_critic": 1e-3,
        "max_training_timesteps": int(3e6),
        "max_ep_len": 1000,
        "update_timestep": 1000 * 4,
        "log_freq": 1000 * 2,
        "print_freq": 2000 * 5,
        "save_model_freq": int(1e5),
    }
    train_ppo(config)


def eval():
    config = {
        "render_mode": None,     # None for faster training, "human" to watch
        "time_limit_steps": 500,
        "gamma": 0.99,
        "eps_clip": 0.2,
        "K_epochs": 4,
        "lr_actor": 1e-4,
        "lr_critic": 1e-3,
        "max_training_timesteps": int(3e6),
        "max_ep_len": 1000,
        "update_timestep": 1000 * 4,
        "log_freq": 1000 * 2,
        "print_freq": 2000 * 5,
        "save_model_freq": int(1e5),
    }

    env = make_env(render_mode=None)
    config['state_dim'] = env.observation_space.shape[0]
    config['action_dim'] = env.action_space.n
    print("obs shape:", env.observation_space.shape[0])
    print("action dim:", env.action_space.n)
    agent = PPO(state_dim=config['state_dim'], action_dim=config['action_dim'], config=config)
    agent.load_safetensors(checkpoint_path="runs\\EscapeRoom\\run_000\\models")
    print("Loaded agent from checkpoint.")
    evaluate_agent_and_save_gifs(agent=agent, make_env=make_env, num_episodes=2, out_dir="eval_gifs", gif_prefix="eval_ep", fps=30, mode="rgb_array")




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=["test-random", "test-scripted", "agent-training", "eval", "test2"],
        help="Which script to run",
    )
    parser.add_argument(
        "--human",
        action="store_true",
        help="Render with a human window (slower).",
    )
    args = parser.parse_args()

    render_mode = "human" if args.human else None

    if args.mode == "test-random":
        test_random(render_mode="human" if args.human else "human")  # testing usually wants view
    elif args.mode == "test-scripted":
        test_scripted(render_mode="human" if args.human else "human")
    elif args.mode == "agent-training":
        agent_training(render_mode=render_mode)
    elif args.mode == "eval":
        eval()
    elif args.mode == "test2":
        test_scripted_hardcoded(render_mode="human" if args.human else "human")


if __name__ == "__main__":
    main()
