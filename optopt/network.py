

import gin
import numpy as np
import tensorflow as tf  # pylint: disable=g-explicit-tensorflow-version-import

from tf_agents.networks import lstm_encoding_network
from tf_agents.networks import network
from tf_agents.specs import tensor_spec
from tf_agents.utils import nest_utils

class Exp_normalization_layer(tf.keras.layers.Layer):
    def __init__(self, moving = 0.995, clip = 1):
        super(Exp_normalization_layer, self).__init__()
        self.momentum = self.add_weight(name = "moving", 
                                    shape = (), 
                                    initializer = 'zeros',
                                    dtype = tf.float32, 
                                    trainable = False)
        self.run_count = self.add_weight(name = "run_count", 
                                    shape = (), 
                                    initializer = 'zeros',
                                    dtype = tf.float32, 
                                    trainable = False)
        self.moving = moving
        self.clip = clip

    def build(self, input_shape):
        self.exp_moving_mean = self.add_weight("exp_moving_mean",
                                    shape=[input_shape[-1]], 
                                    initializer = 'zeros',
                                    dtype = tf.float32, 
                                    trainable = False)
        self.exp_moving_var = self.add_weight("exp_moving_mean",
                                    shape=[input_shape[-1]], 
                                    initializer = 'ones',
                                    dtype = tf.float32, 
                                    trainable = False)
        
    def call(self, inputs, training = None):
        # https://stats.stackexchange.com/a/111912
        #최초 샘플이 없기 때문에 mean에 신규 데이터를 포함해서 산정
        tf.print("training = ", training)
        if not training:
            self.run_count.assign_add(1.)
            self.momentum.assign(tf.maximum(1 - 1/self.run_count, self.moving))
            
            var = tf.reduce_mean((inputs - self.exp_moving_mean) ** 2, tf.range(tf.rank(inputs) - 1))
            self.exp_moving_var.assign( self.momentum * (self.exp_moving_var + (1 - self.momentum) * var) )

            mean = tf.reduce_mean(inputs, tf.range(tf.rank(inputs) - 1))
            self.exp_moving_mean.assign((self.exp_moving_mean * self.momentum) + (1 - self.momentum) * mean)

        return tf.clip_by_value( (inputs - self.exp_moving_mean) / tf.maximum(1e-6, self.exp_moving_var ** 0.5), -self.clip, self.clip)

    def get_config(self):
        return {"moving": self.moving, 'clip': self.clip}
class actor_deterministic_rnn_network(network.Network):
  """Creates a recurrent actor network."""
  def __init__(self,
               input_tensor_spec,
               output_tensor_spec,
               preprocessing_layers=None,
               preprocessing_combiner=None,
               conv_layer_params=None,
               input_fc_layer_params=(200, 100),
               input_dropout_layer_params=None,
               lstm_size=None,
               output_fc_layer_params=(200, 100),
               activation_fn=tf.keras.activations.relu,
               dtype=tf.float32,
               rnn_construction_fn=None,
               rnn_construction_kwargs={},
               name='ActorDeterministicRnnNetwork'):
    """Creates an instance of `ActorDistributionRnnNetwork`.
    Args:
      input_tensor_spec: A nest of `tensor_spec.TensorSpec` representing the
        input.
      output_tensor_spec: A nest of `tensor_spec.BoundedTensorSpec` representing
        the output.
      preprocessing_layers: (Optional.) A nest of `tf.keras.layers.Layer`
        representing preprocessing for the different observations.
        All of these layers must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      preprocessing_combiner: (Optional.) A keras layer that takes a flat list
        of tensors and combines them. Good options include
        `tf.keras.layers.Add` and `tf.keras.layers.Concatenate(axis=-1)`.
        This layer must not be already built. For more details see
        the documentation of `networks.EncodingNetwork`.
      conv_layer_params: Optional list of convolution layers parameters, where
        each item is a length-three tuple indicating (filters, kernel_size,
        stride).
      input_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied before
        the LSTM cell.
      input_dropout_layer_params: Optional list of dropout layer parameters,
        each item is the fraction of input units to drop or a dictionary of
        parameters according to the keras.Dropout documentation. The additional
        parameter `permanent`, if set to True, allows to apply dropout at
        inference for approximated Bayesian inference. The dropout layers are
        interleaved with the fully connected layers; there is a dropout layer
        after each fully connected layer, except if the entry in the list is
        None. This list must have the same length of input_fc_layer_params, or
        be None.
      lstm_size: An iterable of ints specifying the LSTM cell sizes to use.
      output_fc_layer_params: Optional list of fully_connected parameters, where
        each item is the number of units in the layer. This is applied after the
        LSTM cell.
      activation_fn: Activation function, e.g. tf.nn.relu, slim.leaky_relu, ...
      dtype: The dtype to use by the convolution and fully connected layers.
      rnn_construction_fn: (Optional.) Alternate RNN construction function, e.g.
        tf.keras.layers.LSTM, tf.keras.layers.CuDNNLSTM. It is invalid to
        provide both rnn_construction_fn and lstm_size.
      rnn_construction_kwargs: (Optional.) Dictionary or arguments to pass to
        rnn_construction_fn.
        The RNN will be constructed via:
        ```
        rnn_layer = rnn_construction_fn(**rnn_construction_kwargs)
        ```
      name: A string representing name of the network.
    Raises:
      ValueError: If 'input_dropout_layer_params' is not None.
    """
    if input_dropout_layer_params:
      raise ValueError('Dropout layer is not supported.')

    self._lstm_encoder = lstm_encoder = lstm_encoding_network.LSTMEncodingNetwork(
        input_tensor_spec=input_tensor_spec,
        preprocessing_layers=preprocessing_layers,
        preprocessing_combiner=preprocessing_combiner,
        conv_layer_params=conv_layer_params,
        input_fc_layer_params=input_fc_layer_params,
        lstm_size=lstm_size,
        output_fc_layer_params=output_fc_layer_params,
        activation_fn=activation_fn,
        rnn_construction_fn=rnn_construction_fn,
        rnn_construction_kwargs=rnn_construction_kwargs,
        dtype=dtype,
        name=name)
        
    super(actor_deterministic_rnn_network, self).__init__(
        input_tensor_spec=input_tensor_spec,
        state_spec=lstm_encoder.state_spec,
        name=name)

    self._output_tensor_spec = output_tensor_spec
    
    required_units = np.prod(output_tensor_spec.shape)
    min_v = output_tensor_spec.minimum
    max_v = output_tensor_spec.maximum

    self._projection_networks = tf.keras.Sequential()
    self._projection_networks.add(tf.keras.layers.Dense(required_units))
    self._projection_networks.add(tf.keras.layers.Lambda(lambda X: tf.reshape(X, tf.concat([tf.shape(X)[:-1],tf.convert_to_tensor(output_tensor_spec.shape)], axis = -1))))
    self._projection_networks.add(tf.keras.layers.Lambda((lambda X: tf.keras.activations.sigmoid(X) * (max_v - min_v) + min_v), dtype = dtype))
    #self._projection_networks.add(tf.keras.layers.Lambda(lambda X: tf.reshape(X, tf.shape(X)[:2] + [-1])))
    #self._projection_networks.add(tf.keras.layers.Dense(required_units))
    #self._projection_networks.add(tf.keras.layers.Lambda(lambda X: tf.reshape(X, tf.shape(X)[:2] + list(output_tensor_spec.shape))))
    #self._projection_networks.add(tf.keras.layers.Lambda((lambda X: tf.keras.activations.sigmoid(X) * (max_v - min_v) + min_v), dtype = dtype))
  @property
  def output_tensor_spec(self):
    return self._output_tensor_spec
  def call(self, observation, step_type, network_state=(), training=False):
    def while_collecting(state):
        state = tf.expand_dims(state, axis = -2)
        output_actions = self._projection_networks(state)
        output_actions = tf.squeeze(output_actions, axis = -2)
        return output_actions
    def while_training(state):
        output_actions = self._projection_networks(state)
        return output_actions
    
    state, network_state = self._lstm_encoder(
        observation, step_type=step_type, network_state=network_state,
        training=training)
    return tf.cond(tf.equal(tf.rank(state), 2), lambda : while_collecting(state), lambda : while_training(state) ), network_state
class weight_metrics_wrapper(tf.keras.models.Model):
    def __init__(self, model:tf.keras.models.Model):
        super(weight_metrics_wrapper, self).__init__()
        self.model = model
    def get_metrics(self):
        name_set = {}
        metrics = []
        for lay in self.model.layers:
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
                        name += f"_{name_set[name]}"
                    else: name_set[name] = 0
                    metrics.append({'value':tf.norm(W, 2), 'name': name})
                except: pass
        return metrics

    def get_metric_names(self):
        return [M['name'] for M in self.get_metrics()]

    def call(self, input, training = None):
        output = self.model(input, training)
        if training:
            for M in self.get_metrics(): self.add_metric(M['value'], name = M['name'])
        return output