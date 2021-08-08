import math
import tensorflow as tf
import warnings
import numpy as np
import threading
from queue import Queue
from typing import List, Dict
import optopt
from optopt import env, agent
do_not_provide_feature_name = ['progress', 'objective']
class Manager(optopt.Management_class):
    def __init__(self, using_features:List[str],
                        objective : str = 'val_acc',
                        direction = 'maximize', 
                        config : optopt.Config = optopt.Config()):
                        
        assert direction in ['maximize', 'minimize']

        assert objective not in do_not_provide_feature_name and f"do not use feature name as `{objective}` "
        for I in using_features:
            assert I not in do_not_provide_feature_name and f"do not use feature name as `{I}` "

        self.object_multiplier = {'maximize':1, 'minimize':-1}[direction]

        self.Variables = Variable_definer()

        self.objective = objective
        self.using_features = ['progress'] + using_features
        self.config = config
        self.compiled = False
    def compile(self):
        assert not self.compiled
        self.set_observation_lock = threading.Lock()
        self.get_observation_lock = threading.Lock()
        self.get_observation_lock.acquire()

        self.set_action_lock = threading.Lock()
        self.get_action_lock = threading.Lock()
        self.get_action_lock.acquire()
        
        self.observation_queue = Queue(2)
        self.action_queue = Queue(2)

        self.Variables.freeze()

        self.in_features = len(self.using_features)
        self.out_features = self.Variables.get_param_cnt()

        self.env = env.Env(self, self.in_features, self.out_features, config = self.config)
        
        self.agent = agent.Agent(self, self.env, config = self.config)
        self.agent_started = False

        self.train_wait_new = True
        self.compiled = True
    def get_callback(self):
        assert self.compiled
        return simple_callback(self, self.using_features, self.objective)
        return [simple_callback(self, self.using_features, self.objective) for _ in range(self.config.parallel_env_cnt)]

    def set_observation(self, infos):#obs_info, rew, done, step_type
        assert len(infos) == 4
        self.set_observation_lock.acquire()
        print("set_observation1", self.set_observation_lock.locked())
        assert self.get_observation_lock.locked()
        print("set_observation2", self.observation_queue.qsize())
        self.observation_queue.put(infos)
        print("set_observation3", self.observation_queue.qsize())
        self.get_observation_lock.release()
    def get_observation(self):
        self.get_observation_lock.acquire()
        print("get_observation1", self.set_observation_lock.locked())
        assert self.set_observation_lock.locked()
        print("get_observation2", self.observation_queue.qsize())
        RET = self.observation_queue.get()
        print("get_observation3", self.observation_queue.qsize())
        self.set_observation_lock.release()
        return RET

    def set_action(self, action):
        self.set_action_lock.acquire()
        print("set_action1", self.get_action_lock.locked())
        assert self.get_action_lock.locked()
        print("set_action2", self.action_queue.qsize())
        self.action_queue.put(action)
        print("set_action3", self.action_queue.qsize())
        self.get_action_lock.release()
    def get_action(self):
        self.get_action_lock.acquire()
        print("get_action1", self.set_action_lock.locked())
        assert self.set_action_lock.locked()
        print("get_action2", self.action_queue.qsize())
        RET = self.action_queue.get()
        print("get_action3", self.action_queue.qsize())
        self.set_action_lock.release()
        return RET
    
    def set_hyperparameters(self):
        action = self.get_action()
        self.Variables.set_values(action)
    
    def train_begin(self):
        assert self.compiled
        assert self.train_wait_new
        self.train_wait_new = False
        self.last_objective = None
        self.set_observation((np.zeros([self.in_features], dtype = self.config.dtype), 0, False, 0))
        if not self.agent_started:
            self.agent_started = True
            def agent_processing(agent):
                agent.prepare()
                agent.start()
            self.agent_thread = threading.Thread(target = agent_processing, args = (self.agent, ))
            self.agent_thread.start()
        self.set_hyperparameters()

    def epoch_end(self, obs_info, obj, done):
        assert self.compiled
        Rew = obj if self.last_objective is None else (obj - self.last_objective)
        step_type = 2 if done else 1
        self.last_objective = obj
        obs = list(zip(*sorted(obs_info.items())))[1]
        self.set_observation((np.asarray(obs, dtype = self.config.dtype), Rew, done, step_type))
        if done: self.train_wait_new = True 
        else: self.set_hyperparameters()



class Variable_definer:
    def __init__(self):
        self.hyper_parameters = {}
        self.is_frozen = False
    def freeze(self):
        self.is_frozen = True
        self.hyper_parameters_names = sorted(list(self.hyper_parameters.keys()))
        self.hyper_parameters = [self.hyper_parameters[K] for K in self.hyper_parameters_names]
    def set_function(self, name, func):
        assert not self.is_frozen
        default_value = func(0.5)
        tfv = tf.Variable(default_value, trainable=False)
        if name in self.hyper_parameters:
            warnings.warn(f"{name} is duplicated, check configration. We apply only first setting.", UserWarning)
        else: self.hyper_parameters[name] = [tfv, func]
        return tfv
    def loguniform(self, name :str , min_v :float, max_v :float):
        assert not self.is_frozen
        assert 0 < min_v < max_v
        min_lv, max_lv = math.log(min_v), math.log(max_v)
        return self.set_function(name, lambda rate: math.exp( (max_lv - min_lv) * rate + min_lv ))
    def uniform(self, name :str , min_v :float = 0., max_v :float = 1.):
        assert not self.is_frozen
        assert min_v < max_v
        return self.set_function(name, lambda rate: ( (max_v - min_v) * rate + min_v ))
    def custom(self, name, func):
        assert not self.is_frozen
        return self.set_function(name, func)
    def get_param_names(self): return self.hyper_parameters_names
    def get_param_cnt(self):   return len(self.hyper_parameters)
    def set_values(self, values : list):
        assert self.is_frozen
        assert len(values) == self.get_param_cnt()
        for [V, func], new_V in zip(self.hyper_parameters, values):
            V.assign(func(new_V))
        
class simple_callback(tf.keras.callbacks.Callback):
    def __init__(self, parent_callback : Manager, using_features, objective):
        self.parent_callback = parent_callback
        self.using_features = using_features
        self.objective = objective
    def set_params(self, params):
        self.epochs = params['epochs']
    def get_info(self, epoch, logs):
        tmp ={'progress':(1 + epoch)/self.epochs}
        tmp.update({K:logs[K] for K in self.using_features if K in logs})
        return tmp, logs[self.objective], (self.epochs == epoch + 1)
    def on_train_begin(self, logs = None):
        self.parent_callback.train_begin()
    def on_epoch_end(self, epoch, logs=None):
        #epoch = 0 으로 시작한다.
        obs, obj, done = self.get_info(epoch, logs)
        self.parent_callback.epoch_end(obs, obj, done)
