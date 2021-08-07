#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  7 13:37:10 2021

@author: map
"""
import tensorflow as tf
import asyncio
from adabelief_tf import AdaBeliefOptimizer

import tf_agents
import os
import reverb
import tempfile
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.sac import tanh_normal_projection_network
#from tf_agents.environments import suite_pybullet
from tf_agents.metrics import py_metrics
from tf_agents.networks import actor_distribution_network
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

tempdir = tempfile.gettempdir()


# Use "num_iterations = 1e6" for better results (2 hrs)
# 1e5 is just so this doesn't take too long (1 hr)
num_iterations = 100000 # @param {type:"integer"}

initial_collect_steps = 10000 # @param {type:"integer"}
collect_steps_per_iteration = 1 # @param {type:"integer"}
replay_buffer_capacity = 10000 # @param {type:"integer"}

batch_size = 256 # @param {type:"integer"}

critic_learning_rate = 3e-4 # @param {type:"number"}
actor_learning_rate = 3e-4 # @param {type:"number"}
alpha_learning_rate = 3e-4 # @param {type:"number"}
target_update_tau = 0.005 # @param {type:"number"}
target_update_period = 1 # @param {type:"number"}
gamma = 0.99 # @param {type:"number"}
reward_scale_factor = 1.0 # @param {type:"number"}

log_interval = 1 # @param {type:"integer"}

num_eval_episodes = 20 # @param {type:"integer"}
eval_interval = 10000 # @param {type:"integer"}

policy_save_interval = 5000 # @param {type:"integer"}

class async_Agent:
    def __init__(self, manager, environment:optopt.ENV, strategy = strategy_utils.get_strategy(tpu=False, use_gpu=False)):
        self.manager = manager
        self.env = environment

        collect_env = eval_env = self.env
        with strategy.scope():
            """
            optimizer = AdaBeliefOptimizer(learning_rate=0.0003,
                                            beta_1=0.9,
                                            beta_2=0.999,
                                            epsilon=1e-9,
                                            weight_decay=0.0,
                                            rectify=True,
                                            amsgrad=False,
                                            sma_threshold=5.0,
                                            print_change_log=False)
            self.agent = tf_agents.agents.PPOAgent(
                self.env.time_step_spec(),
                self.env.action_spec(),
                num_epochs = 5, 
                **params)
            """
            params = {}
            model = tf_agents.networks.Sequential(
                [tf.keras.layers.LSTM(128, return_sequences=True), 
                 tf.keras.layers.LSTM(128, return_sequences=(len(self.env.action_spec().shape) == 2)),
                 tf.keras.layers.Dense(self.env.action_spec().shape[-1], activation = 'sigmoid')]
            )
            params['actor_network'] = model
            model = tf_agents.networks.Sequential(
                [tf.keras.layers.LSTM(128, return_sequences=True), 
                 tf.keras.layers.LSTM(128, return_sequences=(len(self.env.action_spec().shape) == 2)),
                 tf.keras.layers.Dense(1)]
            )
            params['critic_network'] = model

            train_step = train_utils.create_train_step()
            self.tf_agent = tf_agent = sac_agent.SacAgent(
                    self.env.time_step_spec(),
                    self.env.action_spec(),
                    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                        learning_rate=actor_learning_rate),
                    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                        learning_rate=critic_learning_rate),
                    alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                        learning_rate=alpha_learning_rate),
                    target_update_tau=target_update_tau,
                    target_update_period=target_update_period,
                    td_errors_loss_fn=tf.math.squared_difference,
                    gamma=gamma,
                    reward_scale_factor=reward_scale_factor,
                    train_step_counter=train_step,
                    **params)

            tf_agent.initialize()
            
        table_name = 'uniform_table'
        table = reverb.Table(
            table_name,
            max_size=replay_buffer_capacity,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            rate_limiter=reverb.rate_limiters.MinSize(1))

        reverb_server = reverb.Server([table])
        reverb_replay = reverb_replay_buffer.ReverbReplayBuffer(
                            tf_agent.collect_data_spec,
                            sequence_length=2,
                            table_name=table_name,
                            local_server=reverb_server)
        dataset = reverb_replay.as_dataset(
            sample_batch_size=batch_size, num_steps=2).prefetch(50)
        experience_dataset_fn = lambda: dataset
        tf_eval_policy = tf_agent.policy
        eval_policy = py_tf_eager_policy.PyTFEagerPolicy(
                            tf_eval_policy, use_tf_function=True)
        tf_collect_policy = tf_agent.collect_policy
        collect_policy = py_tf_eager_policy.PyTFEagerPolicy(
                            tf_collect_policy, use_tf_function=True)
        random_policy = random_py_policy.RandomPyPolicy(
                            collect_env.time_step_spec(), collect_env.action_spec())
        rb_observer = reverb_utils.ReverbAddTrajectoryObserver(
                            reverb_replay.py_client,
                            table_name,
                            sequence_length=2,
                            stride_length=1)
        initial_collect_actor = actor.Actor(
                            collect_env,
                            random_policy,
                            train_step,
                            steps_per_run=initial_collect_steps,
                            observers=[rb_observer])
        env_step_metric = py_metrics.EnvironmentSteps()
        self.collect_actor = collect_actor = actor.Actor(
                            collect_env,
                            collect_policy,
                            train_step,
                            steps_per_run=1,
                            metrics=actor.collect_metrics(10),
                            summary_dir=os.path.join(tempdir, learner.TRAIN_DIR),
                            observers=[rb_observer, env_step_metric])
        self.eval_actor = eval_actor = actor.Actor(
                            eval_env,
                            eval_policy,
                            train_step,
                            episodes_per_run=num_eval_episodes,
                            metrics=actor.eval_metrics(num_eval_episodes),
                            summary_dir=os.path.join(tempdir, 'eval'),
                            )
        saved_model_dir = os.path.join(tempdir, learner.POLICY_SAVED_MODEL_DIR)

        # Triggers to save the agent's policy checkpoints.
        learning_triggers = [
            triggers.PolicySavedModelTrigger(
                saved_model_dir,
                tf_agent,
                train_step,
                interval=policy_save_interval),
            triggers.StepPerSecondLogTrigger(train_step, interval=1000),
        ]

        self.agent_learner = agent_learner = learner.Learner(
                tempdir,
                train_step,
                tf_agent,
                experience_dataset_fn,
                triggers=learning_triggers)
    def get_eval_metrics(self):
        self.eval_actor.run()
        results = {}
        for metric in self.eval_actor.metrics:
            results[metric.name] = metric.result()
        return results
    def log_eval_metrics(self, step, metrics):
        eval_results = (', ').join('{} = {:.6f}'.format(name, result) for name, result in metrics.items())
        print('step = {0}: {1}'.format(step, eval_results))

    async def training_process(self):
        # Reset the train step
        self.tf_agent.train_step_counter.assign(0)

        # Evaluate the agent's policy once before training.
        avg_return = self.get_eval_metrics()["AverageReturn"]
        self.history = [avg_return]

        for _ in range(num_iterations):
            # Training.
            self.collect_actor.run()
            loss_info = self.agent_learner.run(iterations=1)

            # Evaluating.
            step = self.agent_learner.train_step_numpy

            if eval_interval and step % eval_interval == 0:
                metrics = self.get_eval_metrics()
                self.log_eval_metrics(step, metrics)
                self.history.append(metrics["AverageReturn"])

            if log_interval and step % log_interval == 0:
                print('step = {0}: loss = {1}'.format(step, loss_info.loss.numpy()))

    async def training_process(self):
        # Reset the train step
        self.tf_agent.train_step_counter.assign(0)
        self.history = []

        for _ in range(num_iterations):
            # Training.
            self.collect_actor.run()
            loss_info = self.agent_learner.run(iterations=1)
            self.history = loss_info.loss.numpy()

    async def _start(self):
        asyncio.ensure_future(self.training_process())
    def start(self):
        asyncio.run(self._start())
    def finish(self):
        self.rb_observer.close()
        self.reverb_server.stop()
    def get_history(self):return list(self.history)
import optopt