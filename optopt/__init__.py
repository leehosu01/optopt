
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
class Config:
    def __init__(self, 
                action_first_epochs = True,
                provide_hyperparameter_info = False,
                strategy :tf.distribute.Strategy = None,
                lstm_size = [256],
                replay_buffer_capacity = 10000,
                sequence_length = 5,
                train_batch_size = 32,
                train_iterations = 1,
                parallel_env_cnt = None,
                critic_learning_rate = 3e-4,
                actor_learning_rate = 3e-4,
                alpha_learning_rate = 3e-4,
                target_update_tau = 0.005,
                target_update_period = 1,
                gamma = 1.00 ,
                log_interval = 1,
                collect_episodes_for_training = 1,
                collect_episodes_for_env_testing = 0,
                policy_save_interval = 4,
                savedir = tempfile.gettempdir(),
                dtype = 'float32',
                verbose = 0):
        assert parallel_env_cnt == None# still not implemented!
        if action_first_epochs:
            warnings.warn(f"action_first_epochs = {action_first_epochs}, they act without information", UserWarning)
        self.action_first_epochs = action_first_epochs
        self.provide_hyperparameter_info = provide_hyperparameter_info
        #self.info_dropout = info_dropout
        self.strategy = strategy or tf.distribute.get_strategy()
        self.lstm_size = lstm_size
        self.replay_buffer_capacity = replay_buffer_capacity
        self.sequence_length = sequence_length
        self.train_batch_size = train_batch_size
        self.train_iterations = train_iterations
        self.parallel_env_cnt = parallel_env_cnt
        self.critic_learning_rate =critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.target_update_tau = target_update_tau
        self.target_update_period = target_update_period
        self.gamma = gamma 
        self.log_interval = log_interval
        self.collect_episodes_for_training = collect_episodes_for_training
        self.collect_episodes_for_env_testing = collect_episodes_for_env_testing
        self.policy_save_interval = policy_save_interval
        self.savedir = savedir
        self.dtype = dtype
        self.verbose = verbose
