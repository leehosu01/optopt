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

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts

tf.compat.v1.enable_v2_behavior()


class ENV(py_environment.PyEnvironment):
    
  def __init__(self, action_cnt, feature_cnt):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(action_cnt, ), dtype=np.float32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(None, feature_cnt,), dtype=np.float32, name='observation')
    self.init_obs = np.zeros((feature_cnt), dtype=np.float32)
    self._state = 0
    self._episode_ended = False
    
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec

  def _reset(self):
    self._state = [self.init_obs]
    self._episode_ended = False
    return ts.restart(self._state)

  def _step(self, action):

    if self._episode_ended:
        return self.reset()
    
    if self._episode_ended or self._state >= 21:
        reward = self._state - 21 if self._state <= 21 else -21
        return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
        return ts.transition(
            np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
    """
    # Make sure episodes don't go on forever.
    if action == 1:
      self._episode_ended = True
    elif action == 0:
      new_card = np.random.randint(1, 11)
      self._state += new_card
    else:
      raise ValueError('`action` should be 0 or 1.')

    if self._episode_ended or self._state >= 21:
      reward = self._state - 21 if self._state <= 21 else -21
      return ts.termination(np.array([self._state], dtype=np.int32), reward)
    else:
      return ts.transition(
          np.array([self._state], dtype=np.int32), reward=0.0, discount=1.0)
    """