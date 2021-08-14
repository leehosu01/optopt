import math
import tensorflow as tf
import warnings
import numpy as np
import threading
from queue import Queue
from typing import List, Dict
import optopt
from optopt import env, agent
from typing import Callable
do_not_provide_feature_name = ['progress', 'objective']
def devprint(*args, **kwargs): pass
class Manager(optopt.Management_class):
    def __init__(self, objective : str = 'val_acc',
                        direction = 'maximize', 
                        config : optopt.Config = None):
                        
        assert direction in ['maximize', 'minimize']

        assert objective not in do_not_provide_feature_name and f"do not use feature name as `{objective}` "

        self.object_multiplier = {'maximize':1, 'minimize':-1}[direction]

        self.Variables = env.Variable_definer()

        self.objective = objective
        self.config = config or optopt.Config()
        self.compiled = False
    def compile(self, using_features:List[str]):
        assert not self.compiled

        self.Variables.freeze()
        
        for I in using_features:
            assert I not in do_not_provide_feature_name and f"do not use feature name as `{I}` "
        self.using_features = ['progress'] + using_features
        if self.config.provide_hyperparameter_info:
            self.using_features += self.Variables.get_param_names()
        self.using_features = sorted(self.using_features)

        self.set_observation_lock = threading.Lock()
        self.get_observation_lock = threading.Lock()
        self.get_observation_lock.acquire()

        self.set_action_lock = threading.Lock()
        self.get_action_lock = threading.Lock()
        self.get_action_lock.acquire()
        
        self.observation_queue = Queue(2)
        self.action_queue = Queue(2)

        self.in_features = len(self.using_features)
        self.out_features = self.Variables.get_param_cnt()

        self.env = env.Env(self,
                                in_feature_cnt = self.in_features, 
                                Variable_definer = self.Variables,
                                config = self.config)
        
        self.agent = agent.Agent(self, self.env, config = self.config)
        self.agent_started = False

        self.train_wait_new = True
        self.compiled = True
    def get_callback(self, additional_metrics :List[optopt.Metric_wrapper] = []):
        assert self.compiled
        return simple_callback(self, self.using_features, self.objective, get_additional_metrics = additional_metrics)

    def set_observation(self, infos):#obs_info, rew, done
        assert len(infos) == 3
        self.set_observation_lock.acquire()
        devprint("set_observation1", self.set_observation_lock.locked())
        assert self.get_observation_lock.locked()
        devprint("set_observation2", self.observation_queue.qsize())
        self.observation_queue.put(infos)
        devprint("set_observation3", self.observation_queue.qsize())
        self.get_observation_lock.release()
    def get_observation(self):
        self.get_observation_lock.acquire()
        devprint("get_observation1", self.set_observation_lock.locked())
        assert self.set_observation_lock.locked()
        devprint("get_observation2", self.observation_queue.qsize())
        RET = self.observation_queue.get()
        devprint("get_observation3", self.observation_queue.qsize())
        self.set_observation_lock.release()
        return RET

    def set_action(self, action):
        self.set_action_lock.acquire()
        devprint("set_action1", self.get_action_lock.locked())
        assert self.get_action_lock.locked()
        devprint("set_action2", self.action_queue.qsize())
        self.action_queue.put(action)
        devprint("set_action3", self.action_queue.qsize())
        self.get_action_lock.release()

    def get_action(self):
        self.get_action_lock.acquire()
        devprint("get_action1", self.set_action_lock.locked())
        assert self.set_action_lock.locked()
        devprint("get_action2", self.action_queue.qsize())
        RET = self.action_queue.get()
        devprint("get_action3", self.action_queue.qsize())
        self.set_action_lock.release()
        return RET
    
    def set_hyperparameters(self):
        action = self.get_action()
        if action is None: 
            print("set_hyperparameters: re initialize hyper parameters")
            self.Variables.initialize_values()
        else: self.Variables.shift_values(action)
    
    def train_begin(self, obs_info = {}, done = False):
        assert self.compiled
        assert self.train_wait_new
        self.train_wait_new = False
        self.last_objective = None
        if self.config.action_first_epochs:
            self.set_observation((np.asarray([obs_info.get(K, 0.) for K in self.using_features], dtype = self.config.dtype), 0, done))
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
        obj_delta = obj if self.last_objective is None else (obj - self.last_objective)
        Rew = obj_delta * self.object_multiplier
        self.last_objective = obj
        obs = list(zip(*sorted(obs_info.items())))[1]
        if self.config.provide_hyperparameter_info:
            obs = list(obs) + self.Variables.get_values()
        self.set_observation((np.asarray(obs, dtype = self.config.dtype), Rew, done))
        if done: self.train_wait_new = True 
        else: self.set_hyperparameters()



class simple_callback(tf.keras.callbacks.Callback):
    def __init__(self, parent_callback : Manager, using_features, objective, get_additional_metrics: List[optopt.Metric_wrapper] = None):
        self.parent_callback = parent_callback
        self.using_features = using_features
        self.objective = objective
        self.get_additional_metrics = get_additional_metrics
    def set_params(self, params):
        self.epochs = params['epochs']
    def get_info(self, epoch, logs):
        obj = logs.get(self.objective, None)
        tmp ={'progress':(1 + epoch)/self.epochs}
        tmp.update({K:logs[K] for K in self.using_features if K in logs})
        if self.get_additional_metrics is not None:
            for metric_wrapper in self.get_additional_metrics:
                logs = metric_wrapper.get_metrics()
                tmp.update({K:logs[K] for K in self.using_features if K in logs})
        return tmp, obj, (self.epochs == epoch + 1)
    def on_train_begin(self, logs = None):
        obs, _, done = self.get_info(-1, logs)
        self.parent_callback.train_begin(obs, done)
    def on_epoch_end(self, epoch, logs=None):
        #epoch = 0 으로 시작한다.
        obs, obj, done = self.get_info(epoch, logs)
        self.parent_callback.epoch_end(obs, obj, done)
