import os
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from src.escape_room.utils.logger import logger


def _to_np_obs(obs):
    """Make observation compatible with your agent (handles list/tuple/np/tensor-ish)."""
    if isinstance(obs, np.ndarray):
        return obs
    try:
        return np.asarray(obs)
    except Exception:
        return obs  # fallback (if your agent can handle it)


def evaluate_agent_and_save_gifs(
    agent,
    make_env,
    num_episodes=5,
    out_dir="eval_gifs",
    gif_prefix="eval_ep",
    fps=30,
    mode=None,
    max_steps=None,
    seed=None,
):
    """
    Runs evaluation episodes using agent actions and saves one GIF per episode.

    Requirements:
      - make_env(render_mode="rgb_array") must create env
      - env.render() must return an RGB frame (H, W, 3) uint8
      - agent.select_action(obs) -> action (or (action, logprob, value))
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Create env in rgb_array mode for capturing frames
    env = make_env(render_mode=mode)

    scores = []

    for ep in range(1, num_episodes + 1):
        obs, info = env.reset(seed=(None if seed is None else seed + ep))
        done = False
        score = 0.0
        step = 0

        frames = []

        # Capture initial frame
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        while not done:
            step += 1
            if max_steps is not None and step > max_steps:
                break

            state = _to_np_obs(obs)

            # Your PPO returns (action, action_logprob, state_val)
            sel = agent.select_action(state)
            action = sel[0] if isinstance(sel, (tuple, list)) else sel

            obs, reward, terminated, truncated, info = env.step(action)
            done = bool(terminated or truncated)
            score += float(reward)

            frame = env.render()
            if frame is not None:
                frames.append(frame)

        scores.append(score)

        # Save GIF for this episode
        gif_path = out_dir / f"{gif_prefix}_{ep:02d}.gif"
        if len(frames) > 0:
            imageio.mimsave(gif_path, frames, fps=fps)
            logger.info(f"[OK] Saved GIF: {gif_path} | frames={len(frames)} | score={score:.2f}")
        else:
            logger.warning(f"[WARN] No frames captured; GIF not saved for episode {ep} | score={score:.2f}")

    env.close()

    logger.info(f"\nScores per episode: {[round(s, 2) for s in scores]}")
    logger.info(f"Average score: {round(float(np.mean(scores)), 2)}")
    return scores
