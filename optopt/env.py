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
  def __init__(self, manager :optopt.Management_class, feature_cnt, Variable_definer:optopt.Variable_class, config : optopt.Config):
    self.Variables = Variable_definer
    self._action_spec = array_spec.BoundedArraySpec(
        shape=(self.Variables.get_param_cnt(), ), dtype=np.float32, minimum=0, maximum=1, name='action')
    self._observation_spec = array_spec.BoundedArraySpec(
        shape=(feature_cnt, ), dtype=np.float32, name='observation')
    self._state = 0
    self._episode_ended = False
    self.manager = manager
    self.config = config
    
    self.wait_reset = True
    self.is_reset = False

    self.last_observation = None
    self.steps = None
  def action_spec(self):
    return self._action_spec

  def observation_spec(self):
    return self._observation_spec
  def cast(self, *args):
    return [np.asarray(I, dtype = self.config.dtype) for I in args]
  def _reset(self):
    self.Variables.initialize_values()
    self.steps = 0
    self.last_observation = RET = self.manager.get_observation() if not self.is_reset or self.wait_reset else self.last_observation
    if not self.wait_reset:
      print("ABNORMAL! reset with self.wait_reset == ", self.wait_reset)

    Obs, Rew, self._episode_ended = RET
    print("ENV._reset : ", Obs, Rew, self._episode_ended)
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
    self.last_observation = Obs, Rew, self._episode_ended, step_type = self.manager.get_observation()

    if self._episode_ended:
      self.wait_reset = True
      return ts.termination(Obs, Rew)
    else: return ts.transition(*self.cast(Obs, Rew, 1. - self._episode_ended))

def single_or_two_as_two(X):
    if type(X) == float: X = [X, X]
    if len(X) == 1: X = [X[0], X[0]] 
    assert len(X) == 2
    return X
def interpolation(Rate:float, min_v:float, max_v:float):
    return min_v + Rate * (max_v - min_v)
class Variable_definer(optopt.Variable_class):
    def __init__(self):
        self.hyper_parameters = {}
        self.is_frozen = False
    def freeze(self):
        self.is_frozen = True
        self.hyper_parameters_names = sorted(list(self.hyper_parameters.keys()))
        self.hyper_parameters = [self.hyper_parameters[K] for K in self.hyper_parameters_names]
    def initialize_values(self):
        assert self.is_frozen
        for tfv, init_f, _, _ in self.hyper_parameters:
            tfv.assign(init_f())
    def shift_values(self, values : Union[list, np.ndarray]):
        assert self.is_frozen
        assert len(values) == self.get_param_cnt()
        for [V, _, func, _], v2 in zip(self.hyper_parameters, values):
            V.assign(func(v2))
    def get_values(self):
        assert self.is_frozen
        return [V.value().numpy() for V, _, _, _ in self.hyper_parameters]
    def set_values(self, values : Union[list, np.ndarray]):
        assert self.is_frozen
        assert len(values) == self.get_param_cnt()
        for [V, _, _, func], v2 in zip(self.hyper_parameters, values):
            V.assign(func(v2))
    def uniform(self, name :str, init_v : Union[float, Tuple[float], List[float]],
                                 shift_v : Union[float, Tuple[float], List[float]] = [-.1, .1],
                                 min_v : float = None, max_v : float = None, 
                                 post_processing :Callable = (lambda X:X)):
        init_min, init_max = single_or_two_as_two(init_v)
        shift_min, shift_max = (-shift_v, shift_v) if type(shift_v)== float else single_or_two_as_two(shift_v)
        min_v = min_v or init_min
        max_v = max_v or init_max
        assert shift_min <= shift_max
        assert min_v <= init_min
        assert init_min <= init_max
        assert init_max <= max_v
        
        if name in self.hyper_parameters:
            warnings.warn(f"{name} is duplicated, check configration. We apply only first setting.", UserWarning)
        
        projector = (lambda X:post_processing(X))
        tf_Value = tf.Variable(init_min, trainable=False)
        init_function = (lambda : projector(interpolation(random.random(), min_v, max_v)))
        shift_function= (lambda R: projector(np.clip( tf_Value + interpolation(R, shift_min, shift_max) , min_v, max_v) ))
        compat_function= (lambda R: projector(interpolation(R, min_v, max_v) ))
        self.hyper_parameters[name] = [tf_Value, init_function, shift_function, compat_function]

        return tf_Value
    def loguniform(self, name :str, init_v : Union[float, Tuple[float], List[float]],
                                 shift_v : Union[float, Tuple[float], List[float]] = [.5, 2.],
                                 min_v : float = None, max_v : float = None,
                                 post_processing :Callable = (lambda X:X)):
        init_min, init_max = single_or_two_as_two(init_v)
        shift_min, shift_max = (1/shift_v, shift_v) if type(shift_v)== float else single_or_two_as_two(shift_v)
        min_v = min_v or init_min
        max_v = max_v or init_max
        assert 0 < shift_min <= shift_max
        assert 0 < min_v <= init_min
        assert 0 < init_min <= init_max
        assert 0 < init_max <= max_v
        
        if name in self.hyper_parameters:
            warnings.warn(f"{name} is duplicated, check configration. We apply only first setting.", UserWarning)
        
        init_min = math.log(init_min)
        init_max = math.log(init_max)
        shift_min = math.log(shift_min)
        shift_max = math.log(shift_max)
        min_v = math.log(min_v)
        max_v = math.log(max_v)
        projector = (lambda X:post_processing(math.exp(X)))
        tf_Value = tf.Variable(init_min, trainable=False)
        init_function = (lambda : projector(interpolation(random.random(), min_v, max_v)))
        shift_function= (lambda R: projector(np.clip( tf_Value + interpolation(R, shift_min, shift_max) , min_v, max_v) ))
        compat_function= (lambda R: projector(interpolation(R, min_v, max_v) ))
        self.hyper_parameters[name] = [tf_Value, init_function, shift_function, compat_function]

        return tf_Value