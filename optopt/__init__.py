from hashlib import new
import math 
import tensorflow as tf
import warnings
import random

do_not_provide_feature_name = ['progress', 'objective']
class OPT:
    def __init__(self, using_features:list, objective : str = 'val_acc', direction = 'maximize'):
        assert direction in ['maximize', 'minimize']

        assert objective not in do_not_provide_feature_name and f"do not use feature name as `{objective}` "
        for I in using_features:
            assert I not in do_not_provide_feature_name and f"do not use feature name as `{I}` "

        self.object_multiplier = {'maximize':1, 'minimize':-1}[direction]

        self.variable = Variable_definer()

        self.objective = objective
        self.using_features = using_features

        self.callback_logs = {}
        self.compiled = False

    def compile(self):
        self.compiled = True
        self.variable.make_frozen()
        self.A_model = 
        self.V_model = 

    def get_callback(self):
        while 1:
            new_id = random.randint(0, 1000000000)
            if new_id not in self.callback_logs: break
        self.callback_logs[new_id] = []# 이후 pandas Dataframe으로 변환시, col을 정의해둔다
        return simple_callback(new_id, self, self.using_features, self.objective)
        
    def train_begin(self, call_id):
        assert len( self.callback_logs[call_id] ) == 0
        assert self.compiled
    
    def write_log(self, call_id, info):# 이후 pandas Dataframe으로 변환 할 것임
        self.callback_logs[call_id].append(info)
    def read_log(self, call_id):
        return self.callback_logs[call_id]

    def update_model(self, call_id):
        S = self.read_log(call_id) # 관측 ( = 기록)
        V = self.read_log(call_id)[-1][self.objective] * self.object_multiplier # 현재 점수
        #tf-agent 호출 방식 확정 후 정해질 예정
    def setting_value(self, call_id):
        #모델 처리방식 확정 후 정해질 예정

        model_outputs = self.A_model("" , training = False)#inputs
        model_last_output = model_outputs[-1]
        self.variable.set_values(model_last_output)
        pass
    def epoch_end(self, call_id, info):
        self.write_log(call_id, info)
        self.update_model(call_id)
        self.setting_value(call_id)
    def train_end(self, call_id):
        #self.callback_logs[call_id] 를 지워도 되고 상관 없다.
        pass

class Variable_definer:
    def __init__(self):
        self.hyper_parameters = {}
        self.is_frozen = False
    def make_frozen(self):
        self.is_frozen = True
        self.hyper_parameters = sorted(list(self.hyper_parameters), key = lambda K, V: K)
        print("정렬 잘되는지 확인", self.hyper_parameters)
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
    def get_param_cnt(self):
        return len(self.hyper_parameters)
    def set_values(self, values : list):
        assert self.is_frozen
        assert len(values) == self.get_param_cnt()
        for (K, [V, func]), new_V in zip(self.hyper_parameters, values):
            V.assign(func(new_V))
        
class simple_callback(tf.keras.callbacks.Callback):
    def __init__(self, call_id:int, parent_OPT : OPT, using_features, objective):
        self.call_id = call_id
        self.parent_OPT = parent_OPT
        self.using_features = using_features
        self.objective = objective
    def set_params(self, params):
        self.verbose = params['verbose']
        self.epochs = params['epochs']
    def get_info(self, logs):
        tmp ={
            'progress':epoch/self.epochs,
            'objective':logs[self.objective], }
        tmp.update({K:logs[K] for K in self.using_features})
        print("모든 value는 list형 로그가 아닌 단일 value여야함 ", tmp)
        return tmp
    def on_train_begin(self, logs = None):
        self.parent_OPT.epoch_end(self.call_id)
    def on_epoch_end(self, epoch, logs=None):
        self.parent_OPT.epoch_end(self.call_id, self.get_info(logs))
    def on_train_end(self, logs=None):
        self.parent_OPT.train_end(self.call_id)