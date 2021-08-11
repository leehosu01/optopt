#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:37:10 2021

@author: map
"""
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.layers.core import Lambda 
from adabelief_tf import AdaBeliefOptimizer

import tf_agents
import os
import reverb
import tempfile
from tf_agents.agents.ddpg import critic_network, critic_rnn_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
#from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network, actor_distribution_rnn_network
from tf_agents.policies import greedy_policy
from tf_agents.policies import py_tf_eager_policy
from tf_agents.policies import random_py_policy
from tf_agents.replay_buffers import reverb_replay_buffer
from tf_agents.replay_buffers import reverb_utils
from tf_agents.train import actor
from tf_agents.train import learner
from tf_agents.train import triggers
from tf_agents.train.utils import spec_utils
from tf_agents.train.utils import strategy_utils
from tf_agents.train.utils import train_utils

import optopt

import tensorflow_probability as tfp
from tf_agents.networks import network
from tf_agents.distributions import utils as distribution_utils
from tf_agents.keras_layers import bias_layer
from tf_agents.networks import network
from tf_agents.networks import utils as network_utils
from tf_agents.specs import distribution_spec
from tf_agents.specs import tensor_spec

from optopt import network as opt_network
class IdentityProjectionNetwork(network.DistributionNetwork):
  def __init__(self
        , sample_spec
        , mean_activation_fn = tf.keras.activations.tanh
        , std_activation_fn = tf.keras.activations.sigmoid
        , std_scaling = 1. / 16
        , name='NormalProjectionNetwork'):
    if len(tf.nest.flatten(sample_spec)) != 1:
      raise ValueError('Normal Projection network only supports single spec '
                       'samples.')
    output_spec = self._output_distribution_spec(sample_spec, name)
    super(IdentityProjectionNetwork, self).__init__(
        # We don't need these, but base class requires them.
        input_tensor_spec=None,
        state_spec=(),
        output_spec=output_spec,
        name=name)

    self._sample_spec = sample_spec
    self._is_multivariate = sample_spec.shape.ndims > 0
    self._means_projection_layer = tf.keras.layers.Dense(
        sample_spec.shape.num_elements(),
        activation = mean_activation_fn,
        kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01),
        bias_initializer='zeros',
        name='means_projection_layer')

    self._stddev_projection_layer = tf.keras.layers.Dense(
        sample_spec.shape.num_elements(),
        activation=std_activation_fn,
        kernel_initializer='he_normal',
        bias_initializer='zeros',
        name='stddev_projection_layer')
    self.std_scaling = std_scaling
  def _output_distribution_spec(self, sample_spec, network_name):
    is_multivariate = sample_spec.shape.ndims > 0
    input_param_shapes = (
        tfp.distributions.Normal.param_static_shapes(sample_spec.shape))

    input_param_spec = {
        name: tensor_spec.TensorSpec(  # pylint: disable=g-complex-comprehension
            shape=shape,
            dtype=sample_spec.dtype,
            name=network_name + '_' + name)
        for name, shape in input_param_shapes.items()
    }

    def distribution_builder(*args, **kwargs):
      if is_multivariate:
        # For backwards compatibility, and because MVNDiag does not support
        # `param_static_shapes`, even when using MVNDiag the spec
        # continues to use the terms 'loc' and 'scale'.  Here we have to massage
        # the construction to use 'scale' for kwarg 'scale_diag'.  Since they
        # have the same shape and dtype expectationts, this is okay.
        kwargs = kwargs.copy()
        kwargs['scale_diag'] = kwargs['scale']
        del kwargs['scale']
        distribution = tfp.distributions.MultivariateNormalDiag(*args, **kwargs)
      else:
        distribution = tfp.distributions.Normal(*args, **kwargs)
      return distribution

    return distribution_spec.DistributionSpec(
        distribution_builder, input_param_spec, sample_spec=sample_spec)

  def call(self, inputs, outer_rank, training=False, mask=None):
    if inputs.dtype != self._sample_spec.dtype:
      raise ValueError(
          'Inputs to NormalProjectionNetwork must match the sample_spec.dtype.')

    if mask is not None:
      raise NotImplementedError(
          'NormalProjectionNetwork does not yet implement action masking; got '
          'mask={}'.format(mask))

    # outer_rank is needed because the projection is not done on the raw
    # observations so getting the outer rank is hard as there is no spec to
    # compare to.
    batch_squash = network_utils.BatchSquash(outer_rank)
    inputs = batch_squash.flatten(inputs)

    means = self._means_projection_layer(inputs, training=training)
    means = tf.reshape(means, [-1] + self._sample_spec.shape.as_list())

    stds = self._stddev_projection_layer(inputs, training=training) * self.std_scaling
    stds = tf.cast(stds, self._sample_spec.dtype)

    means = batch_squash.unflatten(means)
    stds = batch_squash.unflatten(stds)

    return self.output_spec.build_distribution(loc=means, scale=stds), ()
class Agent(optopt.Agency_class):
    def __init__(self, manager:optopt.Management_class,
                    environment :optopt.Environment_class,
                    config :optopt.Config,
                    strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)):
        self.manager = manager
        self.env = environment
        self.config = config
        self.strategy = strategy
        collect_env = self.env
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_env)
        with self.strategy.scope():
            params = {}
            """
            model = tf_agents.networks.actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                        observation_spec, action_spec, preprocessing_layers=Exp_normalization_layer(clip = 2),
                        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1), conv_layer_params=None, input_fc_layer_params=[]
                        , input_dropout_layer_params=None, lstm_size=[256],
                        output_fc_layer_params=[], activation_fn=tf.keras.activations.relu,
                        dtype=tf.float32, #discrete_projection_net=_categorical_projection_net,
                        continuous_projection_net=tanh_normal_projection_network.TanhNormalProjectionNetwork,#(IdentityProjectionNetwork),
                        rnn_construction_fn=None,
                        rnn_construction_kwargs={}, name='ActorDistributionRnnNetwork'
                    )
            """
            model = opt_network.actor_deterministic_rnn_network(
                        observation_spec, action_spec, preprocessing_layers=opt_network.Exp_normalization_layer(clip = 2),
                        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
                        input_fc_layer_params=[], input_dropout_layer_params=None,
                        lstm_size=self.config.lstm_size,
                        output_fc_layer_params=[], activation_fn=tf.keras.activations.relu,
                        dtype=tf.float32)
            params['actor_network'] = model

            model = tf_agents.networks.value_rnn_network.ValueRnnNetwork(
                        (observation_spec, action_spec), preprocessing_layers=(opt_network.Exp_normalization_layer(clip = 2), tf.keras.layers.Lambda(lambda X:X)),
                        preprocessing_combiner=tf.keras.layers.Concatenate(axis=-1),
                        conv_layer_params=None, input_fc_layer_params=[],
                        input_dropout_layer_params=None, lstm_size=self.config.lstm_size, output_fc_layer_params=[],
                        activation_fn=tf.keras.activations.relu, dtype=tf.float32,
                        name='ValueRnnNetwork'
                    )
            params['critic_network'] = model

            train_step = self.train_step = train_utils.create_train_step()
            self.tf_agent = tf_agent = tf_agents.agents.Td3Agent(
                    time_step_spec,
                    action_spec,
                    actor_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.config.actor_learning_rate),
                    critic_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.config.critic_learning_rate),
                    train_step_counter=train_step,
                    target_update_tau=self.config.target_update_tau,
                    gamma=self.config.gamma,
                    **params)

            tf_agent.initialize()
            
    def get_eval_metrics(self):
        self.eval_actor.run()
        results = {}
        for metric in self.eval_actor.metrics:
            results[metric.name] = metric.result()
        return results
    def log_eval_metrics(self, step, metrics):
        eval_results = (', ').join('{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))

    def prepare(self):
        self.reach_prepare = True
        with self.strategy.scope():
            collect_env = self.env
            tf_agent = self.tf_agent
            train_step = self.train_step

            table_name = 'uniform_table'
            table = reverb.Table(
                table_name,
                max_size=self.config.replay_buffer_capacity,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                rate_limiter=reverb.rate_limiters.MinSize(1))

            reverb_server = reverb.Server([table])
            reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
                                tf_agent.collect_data_spec,
                                sequence_length=self.config.sequence_length,
                                table_name=table_name,
                                local_server=reverb_server)
            dataset = reverb_replay.as_dataset(
                sample_batch_size=self.config.train_batch_size,
                num_steps=self.config.sequence_length )
            _experience_dataset_fn = lambda: dataset
            def experience_dataset_fn():
                print('start training')
                return _experience_dataset_fn()
            
            tf_collect_policy = tf_agent.collect_policy
            collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
                                tf_collect_policy, use_tf_function=True)
            random_policy = random_py_policy.RandomPyPolicy(
                                collect_env.time_step_spec(), collect_env.action_spec())
            

            params = {'pad_end_of_episodes': True} if tf_agents.__version__ >= '0.8.0' else {}
            rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                                reverb_replay.py_client,
                                table_name,
                                sequence_length=self.config.sequence_length,
                                stride_length=1, **params)
            if self.config.collect_episodes_for_env_testing:
                self.initial_collect_actor = actor.Actor(
                                collect_env,
                                random_policy,
                                train_step,
                                episodes_per_run=self.config.collect_episodes_for_env_testing,
                                observers=[rb_observer])
                                
            env_step_metric = py_metrics.EnvironmentSteps()
            self.collect_actor = collect_actor = actor.Actor(
                                collect_env,
                                collect_policy,
                                train_step,
                                episodes_per_run=self.config.collect_episodes_for_training,
                                metrics=actor.collect_metrics(10),
                                summary_dir=os.path.join(self.config.savedir, learner.TRAIN_DIR),
                                observers=[rb_observer, env_step_metric])
                                
            saved_model_dir = os.path.join(self.config.savedir, learner.POLICY_SAVED_MODEL_DIR)

            # Triggers to save the agent's policy checkpoints.
            learning_triggers = [
                triggers.PolicySavedModelTrigger(
                    saved_model_dir,
                    tf_agent,
                    train_step,
                    interval=self.config.policy_save_interval),
                triggers.StepPerSecondLogTrigger(train_step, interval=1000),
            ]

            self.agent_learner = agent_learner = learner.Learner(
                    self.config.savedir,
                    train_step,
                    tf_agent,
                    experience_dataset_fn,
                    triggers=learning_triggers,
                    strategy = self.strategy)
            # Reset the train step
            self.tf_agent.train_step_counter.assign(0)
            self.history = []
            self.finish_prepare = True
    def start(self):
        assert self.reach_prepare
        assert self.finish_prepare
        with self.strategy.scope():
            try: self.initial_collect_actor.run()
            except: pass
            self.reach_start = True
            episode = 0  
            while 1:
                # Training.
                self.collect_actor.run()
                episode += 1

                loss_info = self.agent_learner.run(iterations=int(self.config.train_iterations) )
                self.history.append(loss_info.loss.numpy())
                if self.config.verbose and episode % self.config.verbose == 0:
                    print(np.mean(self.history[-self.config.verbose]))
    def finish(self):
        self.rb_observer.close()
        self.reverb_server.stop()
    def get_history(self):return list(self.history)