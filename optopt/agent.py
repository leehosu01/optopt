#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:37:10 2021

@author: map
"""
import tensorflow as tf
import numpy as np 
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
class Agent(optopt.Agency_class):
    def __init__(self, manager:optopt.Management_class, environment :optopt.Environment_class, config :optopt.Config, strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)):
        self.manager = manager
        self.env = environment
        self.config = config

        collect_env = self.env
        observation_spec, action_spec, time_step_spec = spec_utils.get_tensor_specs(collect_env)
        with strategy.scope():
            params = {}
            model = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
                    observation_spec, action_spec,
                    input_fc_layer_params=[],
                    lstm_size = [128, 128],
                    output_fc_layer_params = [32],
                    activation_fn = tf.keras.activations.relu,
                    continuous_projection_net=(
                        tanh_normal_projection_network.TanhNormalProjectionNetwork))
            params['actor_network'] = model
            model = critic_rnn_network.CriticRnnNetwork(
                        (observation_spec, action_spec), observation_conv_layer_params=None,
                        observation_fc_layer_params=(64,), action_fc_layer_params=(64,),
                        joint_fc_layer_params=[128], lstm_size=[128], output_fc_layer_params=(64, ),
                        activation_fn=tf.keras.activations.relu, kernel_initializer=None,
                        last_kernel_initializer=None, rnn_construction_fn=None,
                        rnn_construction_kwargs=None, name='CriticRnnNetwork'
                    )

            params['critic_network'] = model

            train_step = self.train_step = train_utils.create_train_step()
            self.tf_agent = tf_agent = sac_agent.SacAgent(
                    time_step_spec,
                    action_spec,
                    actor_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.config.actor_learning_rate),
                    critic_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.config.critic_learning_rate),
                    alpha_optimizer=tf.keras.optimizers.Adam(
                        learning_rate=self.config.alpha_learning_rate),
                    target_update_tau=self.config.target_update_tau,
                    target_update_period=self.config.target_update_period,
                    td_errors_loss_fn=tf.math.squared_difference,
                    gamma=self.config.gamma,
                    reward_scale_factor=1.,
                    train_step_counter=train_step,
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
            sample_batch_size=self.config.train_batch_size).prefetch(32)
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
        if self.config.initial_collect_episodes:
            self.initial_collect_actor = actor.Actor(
                            collect_env,
                            random_policy,
                            train_step,
                            episodes_per_run=self.config.initial_collect_episodes,
                            observers=[rb_observer])
                            
        env_step_metric = py_metrics.EnvironmentSteps()
        self.collect_actor = collect_actor = actor.Actor(
                            collect_env,
                            collect_policy,
                            train_step,
                            episodes_per_run=1,
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
                triggers=learning_triggers)
        # Reset the train step
        self.tf_agent.train_step_counter.assign(0)
        self.history = []
        self.finish_prepare = True
    def start(self):
        assert self.reach_prepare
        assert self.finish_prepare
        self.initial_collect_actor.run()
        self.reach_start = True
        episode = 0  
        while 1:
            # Training.
            self.collect_actor.run()
            loss_info = self.agent_learner.run(iterations=1)
            self.history.append(loss_info.loss.numpy())
            episode += 1
            if self.config.verbose and episode % self.config.verbose == 0:
                print(np.mean(self.history[-self.config.verbose]))
    def finish(self):
        self.rb_observer.close()
        self.reverb_server.stop()
    def get_history(self):return list(self.history)