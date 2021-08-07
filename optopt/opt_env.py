#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 23:25:10 2021

@author: map
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import numpy as np
import asyncio

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()
import optopt

class ENV(py_environment.PyEnvironment):
    
  def __init__(self, manager:optopt.OPT, action_cnt, feature_cnt):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(action_cnt, ), dtype=np.float32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(None, feature_cnt,), dtype=np.float32, name='observation')
    self._state = 0
    self._episode_ended = False
    self.manager = manager
    
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    Obs, Rew, self._episode_ended, step_type = asyncio.run(self.manager.get_observation())
    return ts.restart(Obs)

  def _step(self, action):
    if self._episode_ended: return self.reset()
    asyncio.run(self.manager.set_action(action))
    Obs, Rew, self._episode_ended, step_type = asyncio.run(self.manager.get_observation())

    if self._episode_ended: return ts.termination(Obs, Rew)
    return ts.transition(Obs, Rew, discount = 1 - self._episode_ended)