import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import os
import sys

# Requires fix for gymnasium 0.29: replace solver_iter with solver_niter because of mujoco update, otherwise breaks. Should be fixed in gymnasium 1.0.0
# try: hopper-v4, humanoid-v4, half-cheetah_v4
# baseline quadruped: ant-v4
# pterobot: pterobot-v0
env_name = 'pterobot-v0'

# Create the environment without specifying render_mode for training
env = gym.make(env_name)
env = Monitor(env)  # Wrap the environment with a Monitor for accurate reporting
env = DummyVecEnv([lambda: env])  # Wrap it in a DummyVecEnv for compatibility with SB3

# Instantiate the agent (model) with the PPO algorithm
model = PPO("MlpPolicy", env, verbose=1)

# Edit below line
mode = "both" # "train", "eval", "both"
policy = "ppo"
# model_name = f"{policy}_{env_name}"
model_name = f"ppo_humanoid"
if mode == "train" or mode == "both":
    training_rate = 55000/300 # timesteps per second for pterosaur - human is 100_000/300
    mins_to_train = 0.1
    train_timesteps = training_rate * 60 * 60 * (mins_to_train/60)
    print(f'Estimated training time: {(train_timesteps/training_rate)/60:.2f} m')
    model.learn(total_timesteps=train_timesteps)
    model.save(model_name)
    if mode == "train":
        env.close()
        sys.exit()

if mode == "eval" or mode == "both":
    PPO.load(model_name, env=env)

    # Evaluate the trained agent
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=10)
    print(f"Mean reward: {mean_reward} +/- {std_reward}")

    # For visualization, create a new environment instance with render_mode='human'
    env_vis = gym.make(env_name, render_mode='human')
    env_vis = Monitor(env_vis)  # Wrap the environment with a Monitor for accurate reporting
    env_vis = DummyVecEnv([lambda: env_vis])  # Wrap it in a DummyVecEnv for compatibility with SB3

    # Reset the environment for the new visualization instance
    obs = env_vis.reset()
    for _ in range(10000):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env_vis.step(action)

    env_vis.close()
    env.close()