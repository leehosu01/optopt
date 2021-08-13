from logging import warning
import tensorflow as tf
import optopt 
from typing import List, Union

class weight_metrics_wrapper(tf.keras.models.Model, optopt.Metric_wrapper):
    def __init__(self, model:tf.keras.models.Model, exp_momentum : Union[float, List[float]] = [0.9, 0.95, 0.99]):
        super(weight_metrics_wrapper, self).__init__()
        self.sub_model = model
        self.exp_momentum = exp_momentum
        self.set_weights = model.set_weights
        self.get_weights = model.get_weights
        self.get_layer = model.get_layer
        self.load_weights = model.load_weights
        self.summary = model.summary
    def build(self, *args, **kwargs):
        self.metric_requires = []
        name_set = {}
        for lay in self.sub_model.layers:
            for i, W in enumerate(lay.weights):
                try: 
                    lay_name = lay.name
                    try:
                        int(lay_name.split('_')[-1])
                        lay_name = '_'.join(lay_name.split('_')[:-1])
                    except: lay_name = lay_name
                    name = lay_name + f'_weight_{i}_norm'
                    if name in name_set:
                        name_set[name]+=1
                        name = f"{name}_{name_set[name]}"
                    else: name_set[name] = 0
                    
                    self.metric_requires.append([name, W])
                    
                    #value = tf.norm(W, 2)
                    #self.update_metric(f"{name}", value, self.exp_momentum)
                except: pass

        return self.sub_model.build(*args, **kwargs)
    def call(self, input, training = None):
        output = self.sub_model(input, training)
        if training:
            for name, W in self.metric_requires:
                value = tf.norm(W, 2)
                self.update_metric(f"{name}", value, self.exp_momentum)
        return output
class optimizer_metrics_wrapper(tf.keras.optimizers.Optimizer, optopt.Metric_wrapper):
    def __init__(self
            , sub_optimizer: tf.keras.optimizers.Optimizer
            , momentum_slot_name = 'm'
            , variance_slot_name = 'v'
            , eps_control_as_adabelief = False
            , exp_momentum : Union[float, List[float]] = [0.9, 0.95, 0.99]):
        self.sub_optimizer = sub_optimizer
        self._create_slots = sub_optimizer._create_slots
        self._prepare_local = sub_optimizer._prepare_local
        self._resource_apply_dense = sub_optimizer._resource_apply_dense
        self._resource_apply_sparse = sub_optimizer._resource_apply_sparse
        self.set_weights = sub_optimizer.set_weights
        self.get_weights = sub_optimizer.get_weights
        self.momentum_slot_name = momentum_slot_name
        self.variance_slot_name = variance_slot_name
        self.eps_control_as_adabelief = eps_control_as_adabelief
        self.exp_momentum = exp_momentum
        super(optimizer_metrics_wrapper, self).__init__(name = 'Optimizer_metrics_wrapper')
    def apply_gradients(self,
                        grads_and_vars,
                        name=None,
                        experimental_aggregate_gradients=True):
        def cosnorm(X, Y): return tf.reduce_mean(X * Y) / (tf.norm(X, 2) ** 0.5 * tf.norm(Y, 2) ** 0.5 + 1e-9)
        vars_copy = [tf.identity(vars) for _, vars in grads_and_vars]
        RET = self.sub_optimizer.apply_gradients(
                        grads_and_vars = grads_and_vars,
                        name=name,
                        experimental_aggregate_gradients=experimental_aggregate_gradients)
        try: epsilon = self.sub_optimizer.epsilon
        except: warning.warn(f"optimizer do not have epsilon", UserWarning)
        for [grad, vars], old_vars in zip(grads_and_vars, vars_copy):
            try: momentum = self.sub_optimizer.get_slot(vars, self.momentum_slot_name)
            except: warning.warn(f"optimizer do not have 'm' (a.k.a momentum) for variable name {var.name}", UserWarning)
            try: variance = self.sub_optimizer.get_slot(vars, self.variance_slot_name)
            except: warning.warn(f"optimizer do not have 'v' (a.k.a variance) for variable name {var.name}", UserWarning)
            old_vars *= -1
            old_vars += vars
            update = old_vars
            del old_vars
            try: # inner feature 2
                if self.eps_control_as_adabelief:
                    std_not_less_than_eps = 1. - tf.reduce_mean( tf.less((variance - epsilon) ** 0.5, epsilon) )
                    self.update_metric(f'std_not_less_than_eps_{vars.name}', std_not_less_than_eps, self.exp_momentum)
                else:
                    std_not_less_than_eps = 1. - tf.reduce_mean( tf.less(variance ** 0.5, epsilon) )
                    self.update_metric(f'std_not_less_than_eps_{vars.name}', std_not_less_than_eps, self.exp_momentum)
            except: pass 
            try: # inner feature 3
                # what is pre-LR?? 모르니까 그냥 평균 업데이트 취급함.
                Average_per_parameter_update_magnitude = tf.reduce_mean(update)
                self.update_metric(f'Average_per_parameter_update_magnitude_{vars.name}', Average_per_parameter_update_magnitude, self.exp_momentum)
            except: pass 
            try: # inner feature 5
                Log_ratio_of_update_norm_and_parameter_norm = tf.math.log(tf.norm(update, 2) / ( 1e-9 + tf.norm(vars, 2)))
                self.update_metric(f'Log_ratio_of_update_norm_and_parameter_norm_{vars.name}', Log_ratio_of_update_norm_and_parameter_norm, self.exp_momentum)
            except: pass 
            try: # inner feature 8
                Cosine_similarity_of_gradient_and_momentum = cosnorm(grad, momentum)
                self.update_metric(f'Cosine_similarity_of_gradient_and_momentum_{vars.name}', Cosine_similarity_of_gradient_and_momentum, self.exp_momentum)
            except: pass 
            try: # inner feature 10
                Cosine_similarity_of_gradient_and_update = cosnorm(grad, update)
                self.update_metric(f'Cosine_similarity_of_gradient_and_update_{vars.name}', Cosine_similarity_of_gradient_and_update, self.exp_momentum)
            except: pass 
            try: # inner feature 12
                Cosine_similarity_of_gradient_and_parameter = cosnorm(grad, vars)
                self.update_metric(f'Cosine_similarity_of_gradient_and_parameter_{vars.name}', Cosine_similarity_of_gradient_and_parameter, self.exp_momentum)
            except: pass 
        return RET
