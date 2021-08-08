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
from numpy.core.defchararray import asarray
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

import optopt
class Env(optopt.Environment_class):
    
  def __init__(self, manager :optopt.Management_class, action_cnt, feature_cnt, config : optopt.Config):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(action_cnt, ), dtype=np.float32, minimum=-1, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(feature_cnt, ), dtype=np.float32, name='observation')
    self._state = 0
    self._episode_ended = False
    self.manager = manager
    self.config = config
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec
  def cast(self, *args):
    return [np.asarray(I, dtype = self.config.dtype) for I in args]
  def _reset(self):
    Obs, Rew, self._episode_ended, step_type = self.manager.get_observation()
    print("ENV._reset : ", Obs, Rew, self._episode_ended, step_type)
    return ts.restart(*self.cast(Obs))

  def _step(self, action):
    print("ENV._step", action)
    if self._episode_ended: return self.reset()
    action = (action + 1)/2
    print("ENV => call set_action", action)
    self.recive_action = action##
    self.manager.set_action(action)
    print("ENV <= return set_action")
    Obs, Rew, self._episode_ended, step_type = self.manager.get_observation()

    if self._episode_ended: return ts.termination(Obs, Rew)
    return ts.transition(*self.cast(Obs, Rew, 1. - self._episode_ended))