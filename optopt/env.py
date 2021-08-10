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
  def __init__(self, manager :optopt.Management_class, feature_cnt, action_cnt, config : optopt.Config):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(action_cnt, ), dtype=np.float32, minimum=-1, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(feature_cnt, ), dtype=np.float32, name='observation')
    self._state = 0
    self._episode_ended = False
    self.manager = manager
    self.config = config
    
    self.wait_reset = True
    self.is_reset = False

    self.last_observation = None
    self.hyper_parameter = None
    self.steps = None
  def random_init_parameter(self):
    return tf.random.uniform(self._action_spec.shape, -1.5, +1.5)
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec
  def cast(self, *args):
    return [np.asarray(I, dtype = self.config.dtype) for I in args]
  def _reset(self):
    self.hyper_parameter = self.random_init_parameter()
    self.steps = 0
    self.last_observation = RET = self.manager.get_observation() if not self.is_reset or self.wait_reset else self.last_observation
    if not self.wait_reset:
      print("ABNORMAL! reset with self.wait_reset == ", self.wait_reset)

    Obs, Rew, self._episode_ended, step_type = RET
    print("ENV._reset : ", Obs, Rew, self._episode_ended, step_type)
    self.is_reset = True
    self.wait_reset = False
    return ts.restart(*self.cast(Obs))
  def _step(self, action):
    #assert all((-1<= action ) & ( action <= 1))
    if self._episode_ended: 
      print("ENV._step => reset with following action ..? ", action)
      return self.reset()
    assert not self.wait_reset
    self.is_reset = False
    self.steps += 1

    self.hyper_parameter += action * (1 + max(0, 5 - self.steps) / 4)
    action = tf.sigmoid(self.hyper_parameter)
    self.hyper_parameter = tf.clip_by_value(self.hyper_parameter, -2, 2)

    print("ENV._step => call set_action", action)
    self.manager.set_action(action)
    print("ENV._step <= return set_action")
    self.last_observation = Obs, Rew, self._episode_ended, step_type = self.manager.get_observation()

    if self._episode_ended:
      self.wait_reset = True
      return ts.termination(Obs, Rew)
    else: return ts.transition(*self.cast(Obs, Rew, 1. - self._episode_ended))