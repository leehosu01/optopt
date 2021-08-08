import asyncio
import math
from numpy.core.defchararray import asarray 
import tensorflow as tf
import warnings
import random
import pandas as pd
import numpy as np
import tf_agents
import nest_asyncio
nest_asyncio.apply()
def devprint(*args, **kwargs):
    print(*args, **kwargs, flush = True)
def run_until(X):
    return asyncio.run(X)
    loop = asyncio.get_event_loop()
    loop.set_debug(True)
    loop.run_until_complete(X)
    loop.close()
do_not_provide_feature_name = ['progress', 'objective']
class OPT:
    """
    단일 모델, 단일 agent, 단일 환경만 다룬다.
    동시 다중 모델, agent, 환경은 나중에 업데이트
    """
    def __init__(self, using_features:list, objective : str = 'val_acc', direction = 'maximize'):
        assert direction in ['maximize', 'minimize']

        assert objective not in do_not_provide_feature_name and f"do not use feature name as `{objective}` "
        for I in using_features:
            assert I not in do_not_provide_feature_name and f"do not use feature name as `{I}` "

        self.object_multiplier = {'maximize':1, 'minimize':-1}[direction]

        self.Variables = Variable_definer()

        self.objective = objective
        self.using_features = ['progress'] + using_features

        self.compiled = False

    async def compile(self):
        devprint("OPT.compile start")
        self.compiled = True

        self.Variables.make_frozen()
        self.normalizer = Normalizer(self.using_features)
        
        devprint("OPT.compile build Logger start")
        self.observe_logger = Logger(self.using_features)
        self.action_logger = Logger(self.Variables.get_param_names())
        self.object_logger = Logger([self.objective])

        self.observation_lock_set = asyncio.Lock()
        self.observation_lock_get = asyncio.Lock()
        devprint("OPT.compile observation_lock_set lock start", self.observation_lock_set)
        await self.observation_lock_set.acquire()
        devprint("OPT.compile observation_lock_set lock end", self.observation_lock_set)

        self.action_lock_set = asyncio.Lock()
        self.action_lock_get = asyncio.Lock()
        await self.action_lock_get.acquire()

        self.train_finish = False

        devprint("OPT.compile env init start")
        self.env = opt_env.ENV(self, self.Variables.get_param_cnt(), self.normalizer.get_param_cnt())
        #self.env = tf_agents.environments.py_environment.PyEnvironment(self.env)
        devprint("OPT.compile agent init start")
        self.agent = opt_agent.async_Agent(self, self.env)
        devprint("OPT.compile agent init finish")
        self.agent.start()
        devprint("OPT.compile agent start finish")


    def get_callback(self):
        assert self.compiled
        devprint("OPT.get_callback")
            
        return simple_callback(self, self.using_features, self.objective)
        
    async def set_observation(self, obs_info, obj, done):
        devprint("OPT.set_observation", obs_info, obj, done)
        #assert self.observation_lock_get.locked()
        await self.observation_lock_set.acquire()
        #assert self.observation_lock_get.locked()
        self.observe_logger.write(obs_info)
        self.object_logger.write([obj])
        self.train_finish = done
        self.observation_lock_get.release()
        #assert not self.observation_lock_get.locked()
    async def get_observation(self): # get이 먼저 발생
        devprint("OPT.get_observation")
        #assert self.observation_lock_set.locked()
        #assert 0
        await self.observation_lock_get.acquire()
        #assert self.observation_lock_set.locked()
        self.get_observation1 = True
        Obs = self.normalizer(self.observe_logger.read().values)
        self.get_observation2 = True
        Done = self.train_finish
        self.get_observation3 = True
        Rew = 0
        self.get_observation4 = True
        step_type = 2 if self.train_finish else 1
        self.get_observation5 = True
        if len(self.object_logger.read()) > 1:
            Rew = self.object_logger.read().iloc[-1].values[0] - self.object_logger.iloc[-2].read().values[0]
        elif len(self.object_logger.read()) == 1:
            Rew = self.object_logger.read().iloc[-1].values[0]
        else: step_type = 0
        RET = [Obs, Rew * self.object_multiplier, Done, step_type]
        RET = [np.asarray(I, dtype = 'float32') for I in RET]
        devprint("OPT.get_observation RET = ", RET)
        self.observation_lock_set.release()
        #assert not self.observation_lock_set.locked()
        return tuple(RET)
    async def set_action(self, action):# set이 먼저 발생
        devprint("OPT.set_action", action)
        await self.action_lock_set.acquire()
        self.action_logger.write(action)
        self.action_lock_get.release()
    async def get_action(self):
        devprint("OPT.get_action")
        await self.action_lock_get.acquire()
        RET = self.action_logger.read().iloc[-1].values
        self.action_lock_set.release()
        return RET

    async def set_hyperparameters(self):
        devprint("OPT.set_hyperparameters")
        action = await self.get_action()
        self.Variables.set_values(action)

    async def train_begin(self):
        #assert 0
        devprint("OPT.train_begin")
        assert self.compiled

        self.train_finish = False
        await self.set_hyperparameters()
    async def epoch_end(self, info):
        devprint("OPT.epoch_end", info)
        await self.set_observation(*info)
        await self.set_hyperparameters()
    async def train_end(self):
        devprint("OPT.train_end")
        self.normalizer.update(self.object_logger.read().values)

        self.observe_logger = Logger(self.using_features)
        self.action_logger = Logger(self.Variables.get_param_names())
        self.object_logger = Logger([self.objective])

        self.observation_lock_set = asyncio.Lock()
        self.observation_lock_get = asyncio.Lock()
        await self.observation_lock_set.acquire()

        self.action_lock_set = asyncio.Lock()
        self.action_lock_get = asyncio.Lock()
        await self.action_lock_get.acquire()
        #self.callback_logs[call_id] 를 지워도 되고 상관 없다.
        pass

class Variable_definer:
    def __init__(self):
        self.hyper_parameters = {}
        self.is_frozen = False
    def make_frozen(self):
        self.is_frozen = True
        self.hyper_parameters_names = sorted(list(self.hyper_parameters.keys()))
        self.hyper_parameters = [self.hyper_parameters[K] for K in self.hyper_parameters_names]
        devprint("정렬 잘되는지 확인", self.hyper_parameters)
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
    def __init__(self, parent_OPT : OPT, using_features, objective):
        self.parent_OPT = parent_OPT
        self.using_features = using_features
        self.objective = objective
    def set_params(self, params):
        self.verbose = params['verbose']
        self.epochs = params['epochs']
    def get_info(self, epoch, logs):
        tmp ={'progress':(1 + epoch)/self.epochs}
        tmp.update({K:logs[K] for K in self.using_features if K in logs})
        devprint("모든 value는 list형 로그가 아닌 단일 value여야함 ", tmp)
        devprint("적어도 한번은 0 progress 조건을 놔둬는걸로 epoch = ", epoch)
        return tmp, logs[self.objective], (self.epochs == epoch + 1)
    def on_train_begin(self, logs = None):
        devprint("simple_callback.on_train_begin", logs)
        devprint("simple_callback.on_train_begin 1", logs)
        run_until(self.parent_OPT.train_begin())
        devprint("simple_callback.on_train_begin 2", logs)
        devprint("simple_callback.on_train_begin 3", logs)
    def on_epoch_end(self, epoch, logs=None):
        #epoch = 0 으로 시작한다.
        devprint("simple_callback.on_epoch_end", logs)
        run_until(self.parent_OPT.epoch_end(self.get_info(epoch, logs)))
    def on_train_end(self, logs=None):
        devprint("simple_callback.on_train_end", logs)
        run_until(self.parent_OPT.train_end())


class Logger:
    def __init__(self, params_name):
        self.params_name = params_name
        self.log = pd.DataFrame(columns=self.params_name, dtype = 'float32')
    def write(self, values):
        assert len(values) == len(self.params_name)
        assert type(values) in [list, tuple, dict]
        if type(values) in [list, tuple]: 
            self.log = self.log.append(dict(zip(self.params_name, values)), ignore_index = True)
        if type(values) == dict: 
            self.log = self.log.append(values, ignore_index = True)
    def read(self, cols = None):
        if cols is not None:
            assert all(c in self.params_name for c in cols)
            return self.log.loc[cols]
        return self.log
class Normalizer:
    def __init__(self, params_name):
        self.params_name = params_name
        self.parameters = self.params_name
    def __call__(self, values):
        devprint("is shape [0, N] ?", values.shape)
        return np.concatenate([np.zeros([1, self.get_param_cnt()], dtype = 'float32'), values], axis = 0) 
    def update(self, values):
        pass
    def get_param_cnt(self):
        return len(self.parameters)
from optopt import opt_env
from optopt import opt_agent