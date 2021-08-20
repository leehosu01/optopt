#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:37:10 2021

@author: map
"""
import traceback
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



from tf_agents.trajectories import time_step as ts
from tf_agents.typing import types
from typing import Optional, Text
from tf_agents.utils import nest_utils
class custom_Td3Agent(tf_agents.agents.Td3Agent):

    def critic_loss(self,
                    time_steps: ts.TimeStep,
                    actions: types.Tensor,
                    next_time_steps: ts.TimeStep,
                    weights: Optional[types.Tensor] = None,
                    training: bool = False) -> types.Tensor:
        """Computes the critic loss for TD3 training.

        Args:
        time_steps: A batch of timesteps.
        actions: A batch of actions.
        next_time_steps: A batch of next timesteps.
        weights: Optional scalar or element-wise (per-batch-entry) importance
            weights.
        training: Whether this loss is being used for training.

        Returns:
        critic_loss: A scalar critic loss.
        """
        with tf.name_scope('critic_loss'):
            target_actions, _ = self._target_actor_network(
                next_time_steps.observation, next_time_steps.step_type,
                training=training)

            # Add gaussian noise to each action before computing target q values
            def add_noise_to_action(action):  # pylint: disable=missing-docstring
                dist = tfp.distributions.Normal(loc=tf.zeros_like(action),
                                                scale=self._target_policy_noise *
                                                tf.ones_like(action))
                noise = dist.sample()
                noise = tf.clip_by_value(noise, -self._target_policy_noise_clip,
                                        self._target_policy_noise_clip)
                return action + noise

            noisy_target_actions = tf.nest.map_structure(add_noise_to_action,
                                                        target_actions)

            # Target q-values are the min of the two networks
            target_q_input_1 = (next_time_steps.observation, noisy_target_actions)
            target_q_values_1, _ = self._target_critic_network_1(
                target_q_input_1,
                next_time_steps.step_type,
                training=False)
            target_q_input_2 = (next_time_steps.observation, noisy_target_actions)
            target_q_values_2, _ = self._target_critic_network_2(
                target_q_input_2,
                next_time_steps.step_type,
                training=False)
            target_q_values = (target_q_values_1 + target_q_values_2) / 2

            td_targets = tf.stop_gradient(
                self._reward_scale_factor * next_time_steps.reward +
                self._gamma * (next_time_steps.discount * target_q_values)) # for relu reward apply

            pred_input_1 = (time_steps.observation, actions)
            pred_td_targets_1, _ = self._critic_network_1(
                pred_input_1, time_steps.step_type, training=training)
            pred_input_2 = (time_steps.observation, actions)
            pred_td_targets_2, _ = self._critic_network_2(
                pred_input_2, time_steps.step_type, training=training)
            pred_td_targets_all = [pred_td_targets_1, pred_td_targets_2]

            if self._debug_summaries:
                tf.compat.v2.summary.histogram(
                    name='td_targets', data=td_targets, step=self.train_step_counter)
                with tf.name_scope('td_targets'):
                    tf.compat.v2.summary.scalar(
                        name='mean',
                        data=tf.reduce_mean(input_tensor=td_targets),
                        step=self.train_step_counter)
                    tf.compat.v2.summary.scalar(
                        name='max',
                        data=tf.reduce_max(input_tensor=td_targets),
                        step=self.train_step_counter)
                    tf.compat.v2.summary.scalar(
                        name='min',
                        data=tf.reduce_min(input_tensor=td_targets),
                        step=self.train_step_counter)

                for td_target_idx in range(2):
                    pred_td_targets = pred_td_targets_all[td_target_idx]
                    td_errors = td_targets - pred_td_targets
                    with tf.name_scope('critic_net_%d' % (td_target_idx + 1)):
                        tf.compat.v2.summary.histogram(
                            name='td_errors', data=td_errors, step=self.train_step_counter)
                        tf.compat.v2.summary.histogram(
                            name='pred_td_targets',
                            data=pred_td_targets,
                            step=self.train_step_counter)
                        with tf.name_scope('td_errors'):
                            tf.compat.v2.summary.scalar(
                                name='mean',
                                data=tf.reduce_mean(input_tensor=td_errors),
                                step=self.train_step_counter)
                            tf.compat.v2.summary.scalar(
                                name='mean_abs',
                                data=tf.reduce_mean(
                                    input_tensor=tf.abs(td_errors)),
                                step=self.train_step_counter)
                            tf.compat.v2.summary.scalar(
                                name='max',
                                data=tf.reduce_max(input_tensor=td_errors),
                                step=self.train_step_counter)
                            tf.compat.v2.summary.scalar(
                                name='min',
                                data=tf.reduce_min(input_tensor=td_errors),
                                step=self.train_step_counter)
                        with tf.name_scope('pred_td_targets'):
                            tf.compat.v2.summary.scalar(
                                name='mean',
                                data=tf.reduce_mean(input_tensor=pred_td_targets),
                                step=self.train_step_counter)
                            tf.compat.v2.summary.scalar(
                                name='max',
                                data=tf.reduce_max(input_tensor=pred_td_targets),
                                step=self.train_step_counter)
                            tf.compat.v2.summary.scalar(
                                name='min',
                                data=tf.reduce_min(input_tensor=pred_td_targets),
                                step=self.train_step_counter)

            critic_loss = (self._td_errors_loss_fn(td_targets, pred_td_targets_1)
                        + self._td_errors_loss_fn(td_targets, pred_td_targets_2))
            if nest_utils.is_batched_nested_tensors(
                    time_steps, self.time_step_spec, num_outer_dims=2):
                # Sum over the time dimension.
                critic_loss = tf.reduce_sum(
                    input_tensor=critic_loss, axis=range(1, critic_loss.shape.rank))

            if weights is not None:
                critic_loss *= weights

            return tf.reduce_mean(input_tensor=critic_loss)


class Agent(optopt.Agency_class):
    def __init__(self, manager:optopt.Management_class,
                    environment :optopt.Environment_class,
                    config :optopt.Config):
        self.manager = manager
        self.env = environment
        self.config = config
        collect_env = self.env
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_env)
    
        params = {}
        model = opt_network.actor_deterministic_standard_network(
            observation_spec, action_spec, 
            units = self.config.network_unit,
            masking_rate = self.config.masking_rate,
            config = self.config,
            name = 'actor_network')
        params['actor_network'] = model
        
        model = opt_network.critic_standard_network(
            (observation_spec, action_spec), 
            units = self.config.network_unit,
            masking_rate = self.config.masking_rate,
            config = self.config,
            name = 'critic_network_1')
        params['critic_network'] = model
        
        model = opt_network.critic_standard_network(
            (observation_spec, action_spec), 
            units = self.config.network_unit,
            masking_rate = self.config.masking_rate,
            config = self.config,
            name = 'critic_network_2')
        params['critic_network_2'] = model
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
        """
        train_step = self.train_step = train_utils.create_train_step()
        self.tf_agent = tf_agent = custom_Td3Agent(
                time_step_spec,
                action_spec,
                actor_optimizer=self.config.actor_optimizer_generate_fn(),
                critic_optimizer=self.config.critic_optimizer_generate_fn(),
                train_step_counter=train_step,

                gamma = self.config.gamma,
                target_update_tau = self.config.target_update_tau,
                exploration_noise_std = self.config.exploration_noise_std,
                target_policy_noise = self.config.target_policy_noise,
                target_policy_noise_clip = self.config.target_policy_noise_clip,
                td_errors_loss_fn = self.config.td_errors_loss_fn,

                **params)

        tf_agent.initialize()
    def prepare(self):
        self.reach_prepare = True
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
            sample_batch_size=self.config.training_batch_size,
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
        if self.config.collect_episodes_random_policy:
            self.initial_collect_actor = actor.Actor(
                            collect_env,
                            random_policy,
                            train_step,
                            episodes_per_run=self.config.collect_episodes_random_policy,
                            observers=[rb_observer])
                            
        env_step_metric = py_metrics.EnvironmentSteps()
        self.collect_actor = collect_actor = actor.Actor(
                            collect_env,
                            collect_policy,
                            train_step,
                            episodes_per_run=self.config.collect_episodes_per_run,
                            metrics=actor.collect_metrics(10),
                            summary_dir=os.path.join(self.config.savedir, learner.TRAIN_DIR),
                            observers=[rb_observer, env_step_metric])
                            
        saved_model_dir = os.path.join(self.config.savedir, learner.POLICY_SAVED_MODEL_DIR)

        # Triggers to save the agent's policy checkpoints.
        learning_triggers = []
        try:
            X = triggers.PolicySavedModelTrigger(
                    saved_model_dir,
                    tf_agent,
                    train_step,
                    interval=self.config.policy_save_interval)
            learning_triggers.append(X)
        except Exception as e:
            print("Save model skipped, error = \n", e, "\n------------------ exception report end", flush = True)
            traceback.print_exc()
            print("\n------------------ traceback report end", flush = True)
        learning_triggers.append(triggers.StepPerSecondLogTrigger(train_step, interval=1000))
        
        self.agent_learner = agent_learner = learner.Learner(
                self.config.savedir,
                train_step,
                tf_agent,
                experience_dataset_fn,
                triggers=learning_triggers)
        # Reset the train step
        self.tf_agent.train_step_counter.assign(0)
        self.history = []
        self.finish_prepare = True
    def start(self):
        assert self.reach_prepare
        assert self.finish_prepare

        try:
            self.initial_collect_actor.run()
            
            loss_info = self.agent_learner.run(iterations=int(self.config.training_steps_after_collect) )
            self.history.append(loss_info.loss.numpy())
        except: pass
        self.reach_start = True
        episode = 0  
        while 1:
            # Training.
            with self.config.strategy.scope():
                self.collect_actor.run()
            episode += 1

            loss_info = self.agent_learner.run(iterations=int(self.config.training_steps_after_collect) )
            self.history.append(loss_info.loss.numpy())
            if self.config.verbose and episode % self.config.verbose == 0:
                print(np.mean(self.history[-self.config.verbose]))
    def finish(self):
        self.rb_observer.close()
        self.reverb_server.stop()
    def get_history(self):return list(self.history)