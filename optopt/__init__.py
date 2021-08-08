
import tempfile

class Config:
    def __init__(self, 
                initial_collect_steps = 10 ,
                collect_steps_per_iteration = 1,
                replay_buffer_capacity = 100,
                sequence_length = 8,
                train_batch_size = 1,
                parallel_env_cnt = None,
                critic_learning_rate = 3e-4,
                actor_learning_rate = 3e-4,
                alpha_learning_rate = 3e-4,
                target_update_tau = 0.005,
                target_update_period = 1,
                gamma = 0.99 ,
                log_interval = 1,
                initial_collect_episodes = 8,
                policy_save_interval = 4,
                savedir = tempfile.gettempdir(),
                dtype = 'float32',
                verbose = 0):
        assert parallel_env_cnt == None# still not implemented!!
        self.initial_collect_steps = initial_collect_steps 
        self.collect_steps_per_iteration = collect_steps_per_iteration
        self.replay_buffer_capacity = replay_buffer_capacity
        self.sequence_length = sequence_length
        self.train_batch_size = train_batch_size
        self.parallel_env_cnt = parallel_env_cnt
        self.critic_learning_rate =critic_learning_rate
        self.actor_learning_rate = actor_learning_rate
        self.alpha_learning_rate = alpha_learning_rate
        self.target_update_tau = target_update_tau
        self.target_update_period = target_update_period
        self.gamma = gamma 
        self.log_interval = log_interval
        self.initial_collect_episodes = initial_collect_episodes
        self.policy_save_interval = policy_save_interval
        self.savedir = savedir
        self.dtype = dtype
        self.verbose = verbose