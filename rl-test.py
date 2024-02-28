import os
import sys
from typing import Optional, Dict, Any
import mujoco
from mujoco import mjx
from mujoco import viewer
import jax
from jax import numpy as jp
# from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from my_brax_utils import get_inference_func
from pterobot import Pterobot

# Requires fix for gymnasium 0.29: replace solver_iter with solver_niter because of mujoco update, otherwise breaks. Should be fixed in gymnasium 1.0.0
# try: hopper-v4, humanoid-v4, half-cheetah_v4
# baseline quadruped: ant-v4
# pterobot: pterobot-v0
env_name = 'pterobot'
envs.register_environment(env_name, Pterobot)

# Edit below line
mode = "eval" # "train", "eval", "both"
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
    model_path = 'policy0.zip'
    params = model.load_params(model_path)

    jit_inference_fn = get_inference_func(model_path, action_size=17, observation_size=356)

    eval_env = envs.get_environment(env_name)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)

    # initialize the state
    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    # grab a trajectory
    n_steps = 50000

    mj_model = eval_env.sys.mj_model
    mj_data = mujoco.MjData(mj_model)

    ctrl = jp.zeros(mj_model.nu)
    with viewer.launch_passive(mj_model, mj_data, key_callback=lambda x: 0) as v:
        while v.is_running():
            act_rng, rng = jax.random.split(rng)

            obs = eval_env._get_obs(mjx.put_data(mj_model, mj_data), ctrl)
            ctrl, _ = jit_inference_fn(obs, act_rng)

            mj_data.ctrl = ctrl
            for _ in range(eval_env._n_frames):
                mujoco.mj_step(mj_model, mj_data)  # Physics step using MuJoCo mj_step.
            v.sync()


            