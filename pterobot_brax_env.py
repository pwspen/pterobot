# ackages for plotting and creating graphics
import time
import itertools
import numpy as np
from typing import Callable, NamedTuple, Optional, Union, List
# Graphics and plotting.
import matplotlib.pyplot as plt

# More legible printing from numpy.
np.set_printoptions(precision=3, suppress=True, linewidth=100)

from datetime import datetime
import functools
from IPython.display import HTML
import jax
from jax import numpy as jp
import numpy as np
from typing import Any, Dict, Sequence, Tuple, Union

from brax import base
from brax import envs
from brax import math
from brax.base import Base, Motion, Transform
from brax.envs.base import Env, PipelineEnv, State
from brax.mjx.base import State as MjxState
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
from brax.io import html, mjcf, model

from matplotlib import pyplot as plt
import mujoco
from mujoco import mjx

import os
import shutil


DEFAULT_CAMERA_CONFIG = {
    "distance": 4.0,
}

xml_path = 'load/pterobot_v0.xml'

class Pterobot(PipelineEnv):
  def __init__(
          self,
          xml_file=xml_path,
          ctrl_cost_weight=0.2,
          forward_reward_weight=2.0,
          vertical_reward_weight=1.0,
          healthy_reward=1.0,
          use_contact_forces=False,
          contact_cost_weight=5e-4,
          terminate_when_unhealthy=True,
          healthy_z_range=(0.15, 10),
          contact_force_range=(-1.0, 1.0),
          reset_noise_scale=0.1,
          exclude_current_positions_from_observation=True,
          **kwargs,
      ):
    
    self._ctrl_cost_weight = ctrl_cost_weight
    self._forward_reward_weight = forward_reward_weight
    self._vertical_reward_weight = vertical_reward_weight
    self._healthy_reward = healthy_reward
    self._use_contact_forces = use_contact_forces
    self._contact_cost_weight = contact_cost_weight
    self._terminate_when_unhealthy = terminate_when_unhealthy
    self._healthy_z_range = healthy_z_range
    self._contact_force_range = contact_force_range
    self._reset_noise_scale = reset_noise_scale
    self._exclude_current_positions_from_observation = exclude_current_positions_from_observation

    self._init_qpos = np.array([0.0440588, 0.00120683, 0.2395, 0.993669, 0.00301708, 0.111906, -0.00953076, 0.665, 1.2888e-05, -0.00385139, 1.10028, -0.147647, -0.523495, 0.81469, 1.05431, -0.169102, -0.576586, 0.846909, -0.0040244, 0.0349449, -0.63767, 0.0471247, 0.0148329, -0.637677])
    self._init_qvel = np.array([-0.01, 0, -0.0014, -0.0041, -0.0028, 0.0035, -0.016, 0, -0.0081, 0.012, -0.0045, 0.039, 0.028, 0.013, -0.0042, 0.044, 0.027, 0.0048, 0, 0.0052, -0.0032, 0, -0.0016])


    mj_model = mujoco.MjModel.from_xml_path(xml_file)
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG
    mj_model.opt.iterations = 6
    mj_model.opt.ls_iterations = 6

    sys = mjcf.load_model(mj_model)

    physics_steps_per_control_step = 5
    kwargs['n_frames'] = kwargs.get(
        'n_frames', physics_steps_per_control_step)
    kwargs['backend'] = 'mjx'

    super().__init__(sys, **kwargs)

  def reset(self, rng: jp.ndarray) -> State:
    """Resets the environment to an initial state."""
    rng, rng1, rng2 = jax.random.split(rng, 3)

    low, hi = -self._reset_noise_scale, self._reset_noise_scale

    # Warning: below draw from user-inputted init qpos and qvel!
    qpos = self._init_qpos + jax.random.uniform(
        rng1, (self.sys.nq,), minval=low, maxval=hi
    )
    qvel = self._init_qvel + jax.random.uniform(
        rng2, (self.sys.nv,), minval=low, maxval=hi
    )

    data = self.pipeline_init(qpos, qvel)

    obs = self._get_obs(data, jp.zeros(self.sys.nu))
    reward, done, zero = jp.zeros(3)
    metrics = {
        'reward_fwd': zero,
        'reward_vert': zero,
        'reward_alive': zero,
        'reward_ctrl': zero,
        'reward_lowvel': zero,
        'x_position': zero,
        'y_position': zero,
        'z_position': zero,
        'distance_from_origin': zero,
        'x_velocity': zero,
        'y_velocity': zero,
        'z_velocity': zero,
    }
    return State(data, obs, reward, done, metrics)

  def step(self, state: State, action: jp.ndarray) -> State:
    """Runs one timestep of the environment's dynamics."""
    data0 = state.pipeline_state
    data = self.pipeline_step(data0, action)

    com_before = data0.subtree_com[1]
    com_after = data.subtree_com[1]
    x_pos, y_pos, z_pos = data.q[0:3]
    velocity = (com_after - com_before) / self.dt
    forward_reward = self._forward_reward_weight * velocity[0] # Used to be velocity[0] instead of x_pos, but it just learned to throw itself forward instead of walking.
    vertical_reward = self._vertical_reward_weight * z_pos

    min_z, max_z = self._healthy_z_range
    is_healthy = jp.where(z_pos < min_z, 0.0, 1.0)
    is_healthy = jp.where(z_pos > max_z, 0.0, is_healthy)
    if self._terminate_when_unhealthy:
      healthy_reward = self._healthy_reward
    else:
      healthy_reward = self._healthy_reward * is_healthy

    ctrl_cost = -self._ctrl_cost_weight * jp.sum(jp.square(action))
    lowvel_cost = -1 / jp.sum(jp.square(velocity))

    obs = self._get_obs(data, action)
    reward = forward_reward + healthy_reward + vertical_reward + ctrl_cost + lowvel_cost
    done = 1.0 - is_healthy if self._terminate_when_unhealthy else 0.0
    state.metrics.update(
        reward_fwd=forward_reward,
        reward_vert=vertical_reward,
        reward_alive=healthy_reward,
        reward_ctrl=ctrl_cost,
        reward_lowvel=lowvel_cost,
        x_position=com_after[0],
        y_position=com_after[1],
        z_position=com_after[2],
        distance_from_origin=jp.linalg.norm(com_after),
        x_velocity=velocity[0],
        y_velocity=velocity[1],
        z_velocity=velocity[2],
    )

    return state.replace(
        pipeline_state=data, obs=obs, reward=reward, done=done
    )

  def _get_obs(
      self, data: mjx.Data, action: jp.ndarray
  ) -> jp.ndarray:
    position = data.qpos
    if self._exclude_current_positions_from_observation:
      position = position[2:]

    # external_contact_forces are excluded
    return jp.concatenate([
        position,
        data.qvel,
        data.cinert[1:].ravel(),
        data.cvel[1:].ravel(),
        data.qfrc_actuator,
    ])