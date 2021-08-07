import asyncio
import math 
import tensorflow as tf
import warnings
import random
import pandas as pd
from optopt import opt_env
from optopt import opt_agent
 
devprint = print
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

    def compile(self):
        self.compiled = True
        self.Variables.make_frozen()
        self.normalizer = Normalizer(self.using_features)

        self.observation_lock_turn = asyncio.Lock()
        self.action_lock_turn = asyncio.Lock()

        self.observation_lock = asyncio.Lock()
        self.action_lock = asyncio.Lock()


        self.env = env.ENV(self.Variables.get_param_cnt(), self.normalizer.get_param_cnt())
        self.agent = opt_agent.agent()
        self.agent.start()


    def get_callback(self):
        assert self.compiled
            
        return simple_callback(self, self.using_features, self.objective)
        
    async def set_observation(self, obs_info, obj, done):
        async with self.observation_lock:
            self.observe_logger.write(obs_info)
            self.object_logger.write([obj])
            self.train_finish = done
            self.observation_lock_turn.release()
    async def get_observation(self): # get이 먼저 발생
        async with self.observation_lock:
            await self.observation_lock_turn.acquire()
            Obs = self.normalizer(self.observe_logger.read().values)
            Done = self.train_finish
            Rew = 0
            if len(self.object_logger.read()) > 1:
                Rew = self.object_logger.iloc[0].values[-1] - self.object_logger.iloc[0].values[-2]
            if len(self.object_logger.read()) == 1:
                Rew = self.object_logger.iloc[0].values[0]
            return Obs, Rew * self.object_multiplier, Done
    async def set_action(self, action):# set이 먼저 발생
        async with self.action_lock:
            assert self.action_lock_turn.locked()
            self.action_logger.write(action)
            self.action_lock.release()
    async def get_action(self):
        async with self.action_lock:
            await self.action_lock_turn.acquire()
            return self.action_logger.read().iloc[-1].values

    async def set_hyperparameters(self):
        action = await self.get_action()
        self.Variables.set_values(action)

    async def train_begin(self):
        assert len( self.callback_logger.log ) == 0
        assert self.compiled

        self.observe_logger = Logger(self.using_features)
        self.action_logger = Logger(self.Variables.get_param_names())
        self.object_logger = Logger([self.objective])
        self.train_finish = False

        await self.action_lock.acquire()

    async def epoch_end(self, info):
        await self.set_observation(*info)
        await self.set_hyperparameters()
    async def train_end(self):
        self.normalizer.update(self.object_logger.read().values)
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
        assert min < max
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
        tmp.update({K:logs[K] for K in self.using_features})
        devprint("모든 value는 list형 로그가 아닌 단일 value여야함 ", tmp)
        devprint("적어도 한번은 0 progress 조건을 놔둬는걸로 epoch = ", epoch)
        return tmp, logs[self.objective], (self.epochs == epoch + 1)
    def on_train_begin(self, logs = None):
        asyncio.run(self.parent_OPT.train_begin())
    def on_epoch_end(self, epoch, logs=None):
        #epoch = 0 으로 시작한다.
        asyncio.run(self.parent_OPT.epoch_end(self.get_info(epoch, logs)))
    def on_train_end(self, logs=None):
        asyncio.run(self.parent_OPT.train_end())


class Logger:
    def __init__(self, params_name):
        self.params_name = params_name
        self.log = pd.DataFrame(columns=self.params_name)
    def write(self, values):
        assert len(values) == len(self.params_name)
        assert type(values) in [list, tuple, dict]
        if type(values) in [list, tuple]: 
            self.log = self.log.append(dict(zip(self.params_name, values)))
        if type(values) == dict: 
            self.log = self.log.append(values)
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
        print("is shape [0, N] ?", values.shape)
        return np.concat([np.zeros([1, self.get_param_cnt()]), values], axis = 0) 
    def update(self, values):
        pass
    def get_param_cnt(self):
        return len(self.parameters)