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
import random, warnings, math

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
from typing import Tuple, List, Union, Callable

class Env(optopt.Environment_class):
  def __init__(self, manager :optopt.Management_class, in_feature_cnt, out_feature_cnt, config : optopt.Config):
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(out_feature_cnt, ), dtype=np.float32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(in_feature_cnt, ), dtype=np.float32, name='observation')
    self._state = 0
    self._episode_ended = False
    self.manager = manager
    self.config = config
    
    self.wait_reset = True
    self.is_reset = False

    self.last_observation = None
    self.steps = None
    self.unused_rew = 0.
  def action_spec(self):
        return self._action_spec

  def observation_spec(self):
    return self._observation_spec
  def cast(self, *args):
    return [np.asarray(I, dtype = self.config.dtype) for I in args]
  def _reset(self):
    self.steps = 0
    if self.wait_reset:
      self.manager.set_action(None)
    self.last_observation = RET = self.manager.get_observation() if not self.is_reset or self.wait_reset else self.last_observation
    if not self.wait_reset:
      print("ABNORMAL! reset with self.wait_reset == ", self.wait_reset)

    Obs, Rew, self._episode_ended = RET
    if self.config.guarantee_env_reward_at_reset: self.unused_rew = Rew
    #print("ENV._reset : ", Obs, Rew, self._episode_ended)
    print("ENV._reset : ", Rew, self._episode_ended)
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


    #print("ENV._step => call set_action", action)
    self.manager.set_action(action)
    #print("ENV._step <= return set_action")
    self.last_observation = Obs, Rew, self._episode_ended = self.manager.get_observation()
    Rew += self.unused_rew
    self.unused_rew = 0.
    
    if self._episode_ended:
      self.wait_reset = True
      return ts.termination(Obs, Rew)
    else: return ts.transition(*self.cast(Obs, Rew, 1. - self._episode_ended))