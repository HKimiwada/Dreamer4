import torch
import json
import gymnasium as gym
import ale_py
from pathlib import Path
from atari_preprocessing import AtariPreprocessing

def make_simple_env(game_id="BreakoutNoFrameskip-v4", size=64):
    env = gym.make(game_id, render_mode="rgb_array", frameskip=1)
    env = AtariPreprocessing(env, screen_size=size)
    return env

def collect_data(num_steps=100000, game_id="BreakoutNoFrameskip-v4", save_dir="data/atari/raw"):
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    
    env = make_simple_env(game_id)
    obs, _ = env.reset()
    
    actions_file = open(save_path / "actions.jsonl", "w")
    print(f"Starting collection of {num_steps} steps for {game_id}...")

    for step in range(num_steps):
        # 100k Benchmark logic: Use random actions for the first 100k
        action = env.action_space.sample() 
        next_obs, reward, done, truncated, info = env.step(action)
        
        # Save frame (as .pt for your tokenizer training)
        # Permute to (C, H, W) to match standard PyTorch vision models
        frame_tensor = torch.from_numpy(obs).permute(2, 0, 1).byte()
        torch.save(frame_tensor, save_path / f"frame_{step:06d}.pt")
        
        # Log transition
        log_entry = {
            "step": step,
            "action": int(action),
            "reward": float(reward),
            "is_terminal": bool(done or truncated)
        }
        actions_file.write(json.dumps(log_entry) + "\n")
        
        obs = next_obs
        if done or truncated:
            obs, _ = env.reset()
            
        if step % 1000 == 0:
            print(f"Step {step}/{num_steps} collected.")

    actions_file.close()
    print(f"Collection complete. Data saved to {save_dir}")

if __name__ == "__main__":
    collect_data()