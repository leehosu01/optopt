
import tempfile
import tensorflow as tf
from tf_agents.environments import py_environment
import warnings
from typing import List, Union
class Management_class:
    def __init__(self):
        pass
class Environment_class(py_environment.PyEnvironment):
    def __init__(self):
        pass
class Agency_class:
    def __init__(self):
        pass
class Variable_class:
    def __init__(self):
        pass
    def get_param_names(self): return self.hyper_parameters_name
    def get_param_cnt(self):   return len(self.hyper_parameters)
class Metric_wrapper:
    def update_metric(self, name, value, momentum:Union[float, List]):
        def _sub(name, value, momentum):
            try:MWM = self.MW_metrics.get(name, None)
            except:
                self.MW_metrics = {}
                MWM = self.MW_metrics.get(name, None)
            if MWM is None:
                MWM = self.MW_metrics[name] = Exp_moving_mean_metric(momentum, name = name)
            MWM.update_state(value)
        if type(momentum) == float: _sub(name, value, momentum)
        else:
            momentums = momentum
            for momentum in momentums:
                _name = f"{name}/{momentum}"
                _sub(_name, value, momentum)

    def get_metrics(self):
        return {K:V.result().numpy() for K, V in self.MW_metrics.items()}
    def get_metrics_names(self):
        return list(self.get_metrics().keys())
class Exp_moving_mean_metric(tf.keras.metrics.Metric):
    def __init__(self, moving = 0.995, name='exp_normalization_metric'):
        super(Exp_moving_mean_metric, self).__init__(name=name)
        self.momentum = tf.Variable(0, name = "moving", 
                                    dtype = tf.float32, trainable = False)
        self.run_count = tf.Variable(0, name = "run_count", 
                                    dtype = tf.float32, trainable = False)
        self.exp_moving_mean = tf.Variable(0, name = "exp_moving_mean",
                                    dtype = tf.float32, trainable = False)
        self.moving = moving

    def update_state(self, value, *args, **kwargs):
        self.run_count.assign_add(1.)
        self.momentum.assign(tf.maximum(1 - 1/self.run_count, self.moving))
        value = tf.cast(value, tf.float32)
        self.exp_moving_mean.assign((self.exp_moving_mean * self.momentum) + (1 - self.momentum) * value)
    def result(self):
        return self.exp_moving_mean
class Exp_moving_std_metric(tf.keras.metrics.Metric):
    
    def __init__(self, moving = 0.995, name='exp_normalization_metric'):
        super(Exp_moving_std_metric, self).__init__(name=name)
        self.momentum = tf.Variable(0, name = "moving", 
                                    dtype = tf.float32, trainable = False)
        self.run_count = tf.Variable(0, name = "run_count", 
                                    dtype = tf.float32, trainable = False)
        self.exp_moving_mean = tf.Variable(0, name = "exp_moving_mean",
                                    initializer = 'zeros',
                                    dtype = tf.float32, trainable = False)
        self.exp_moving_var = tf.Variable(1, name = "exp_moving_var",
                                    dtype = tf.float32, trainable = False)
        self.moving = moving

    def update_state(self, value, *args, **kwargs):
        self.run_count.assign_add(1.)
        self.momentum.assign(tf.maximum(1 - 1/self.run_count, self.moving))
        
        value = tf.cast(value, tf.float32)
        var = ((value - self.exp_moving_mean) ** 2)
        self.exp_moving_var.assign( self.momentum * (self.exp_moving_var + (1 - self.momentum) * var) )

        self.exp_moving_mean.assign((self.exp_moving_mean * self.momentum) + (1 - self.momentum) * value)
    def result(self):
        return self.exp_moving_var ** 0.5
class apply_relu_reward_by_mul(float):
    def __new__(self, value):
        return float.__new__(self, value)
    def __init__(self, value):
        float.__init__(value)
        self.value = value
    def __imul__(self, X):
        raise Exception("do not call __imul__ method")
    def __mul__(self, X):
        print("Successfully applying maximum reward via __mul__")
        return tf.maximum(0., self.value * X)
    def __rmul__(self, X):
        print("Successfully applying maximum reward via __rmul__")
        return tf.maximum(0., self.value * X)
def flood_mae_loss(flood = 0.01):
    def _sub(y_true, y_pred):
        return (tf.abs(tf.abs(y_true - y_pred) - flood)+flood)
        return tf.reduce_mean(tf.abs(tf.abs(y_true - y_pred) - flood)+flood)
    return _sub
def pcc_norm(X):
    except_batch = list(range(1, X.shape.rank))
    X = X - tf.reduce_mean(X, axis = except_batch, keepdims=True)
    return tf.math.l2_normalize(X, axis = except_batch)
def CosineSimilarity_loss(centered = True):
    # important*: batch wise centered
    def _sub(y_true, y_pred):
        except_batch = list(range(1, y_true.shape.rank))
        if centered:
            y_true = pcc_norm(y_true)
            y_pred = pcc_norm(y_pred)
            return tf.reduce_mean(y_true * y_pred, axis = except_batch, keepdims = True)
        y_true = tf.math.l2_normalize(y_true, axis = except_batch)
        y_pred = tf.math.l2_normalize(y_pred, axis = except_batch)
        return tf.reduce_mean(y_true * y_pred, axis = except_batch, keepdims = True)
    return _sub
class Config:
    def __init__(self, 
                # environment interaction setting
                action_first_epochs = False,
                provide_hyperparameter_info = False,
                guarantee_env_reward_at_reset = False,

                # inference strategy 
                strategy :tf.distribute.Strategy = None,

                # network option
                network_unit = 256,
                masking_rate  = 0.,

                # training option
                use_maximum_reward :bool= True,
                gamma :float = 1.00,
                target_update_tau :float = 0.005,
                batchNormalization_option = {"momentum": 0.9, "renorm": True, "renorm_momentum": 0.9},

                collect_episodes_random_policy :int = 8,
                collect_episodes_per_run :int = 1,
                training_steps_after_collect  :int = 1,
                training_batch_size = 128,

                # optimizer option
                critic_optimizer_generate_fn = lambda : tf.keras.optimizers.Adam(0.001),
                actor_optimizer_generate_fn = lambda : tf.keras.optimizers.Adam(0.001),


                #reinforcement TD3_agent option
                exploration_noise_std  = 0.05,
                target_policy_noise = 0.01,
                target_policy_noise_clip = 0.05,
                td_errors_loss_fn = CosineSimilarity_loss(),#flood_mae_loss(flood = 0.01),


                #reinforcement agent backend option
                replay_buffer_capacity = 10000,
                sequence_length = 5,
                policy_save_interval = 4,
                savedir = tempfile.gettempdir(),
                dtype = 'float32',
                verbose = 0):
        # environment interaction setting
        self.action_first_epochs = action_first_epochs
        self.provide_hyperparameter_info = provide_hyperparameter_info
        self.guarantee_env_reward_at_reset = guarantee_env_reward_at_reset

        # inference strategy
        self.strategy = strategy

        # network option
        self.network_unit = network_unit
        self.masking_rate  = masking_rate
        self.batchNormalization_option  = batchNormalization_option

        # training option
        self.gamma = apply_relu_reward_by_mul(gamma) if use_maximum_reward else gamma
        self.target_update_tau = target_update_tau

        assert collect_episodes_random_policy > 0
        self.collect_episodes_random_policy = collect_episodes_random_policy
        self.collect_episodes_per_run = collect_episodes_per_run
        self.training_steps_after_collect  = training_steps_after_collect
        self.training_batch_size = training_batch_size

        # optimizer option
        self.critic_optimizer_generate_fn = critic_optimizer_generate_fn
        self.actor_optimizer_generate_fn = actor_optimizer_generate_fn
        
        #reinforcement TD3_agent option
        self.exploration_noise_std = exploration_noise_std
        self.target_policy_noise = target_policy_noise
        self.target_policy_noise_clip = target_policy_noise_clip
        self.td_errors_loss_fn = td_errors_loss_fn 


        #reinforcement agent backend option
        self.replay_buffer_capacity = replay_buffer_capacity
        self.sequence_length = sequence_length
        self.policy_save_interval = policy_save_interval
        self.savedir = savedir
        self.dtype = dtype
        self.verbose = verbose
