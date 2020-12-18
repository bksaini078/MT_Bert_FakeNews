# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Core Keras layers.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import functools
import operator
import sys
import textwrap
import types as python_types
import warnings

import numpy as np
import tensorflow as tf
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.keras.engine.base_layer import Layer
# from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.layers.ops import core as core_ops
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import generic_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, init_ops, standard_ops, gen_math_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import variable_scope
from tensorflow.python.platform import tf_logging
from tensorflow.python.training.tracking import base as trackable
from tensorflow.python.util import dispatch
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import get_canonical_name_for_symbol
from tensorflow.python.util.tf_export import get_symbol_from_name
from tensorflow.python.util.tf_export import keras_export

@keras_export('keras.layers.Dense')
class Dense(Layer):
  """Just your regular densely-connected NN layer.
  `Dense` implements the operation:
  `output = activation(dot(input, kernel) + bias)`
  where `activation` is the element-wise activation function
  passed as the `activation` argument, `kernel` is a weights matrix
  created by the layer, and `bias` is a bias vector created by the layer
  (only applicable if `use_bias` is `True`).
  Note: If the input to the layer has a rank greater than 2, then `Dense`
  computes the dot product between the `inputs` and the `kernel` along the
  last axis of the `inputs` and axis 1 of the `kernel` (using `tf.tensordot`).
  For example, if input has dimensions `(batch_size, d0, d1)`,
  then we create a `kernel` with shape `(d1, units)`, and the `kernel` operates
  along axis 2 of the `input`, on every sub-tensor of shape `(1, 1, d1)`
  (there are `batch_size * d0` such sub-tensors).
  The output in this case will have shape `(batch_size, d0, units)`.
  Besides, layer attributes cannot be modified after the layer has been called
  once (except the `trainable` attribute).
  Example:
  >>> # Create a `Sequential` model and add a Dense layer as the first layer.
  >>> model = tf.keras.models.Sequential()
  >>> model.add(tf.keras.Input(shape=(16,)))
  >>> model.add(tf.keras.layers.Dense(32, activation='relu'))
  >>> # Now the model will take as input arrays of shape (None, 16)
  >>> # and output arrays of shape (None, 32).
  >>> # Note that after the first layer, you don't need to specify
  >>> # the size of the input anymore:
  >>> model.add(tf.keras.layers.Dense(32))
  >>> model.output_shape
  (None, 32)
  Arguments:
    units: Positive integer, dimensionality of the output space.
    activation: Activation function to use.
      If you don't specify anything, no activation is applied
      (ie. "linear" activation: `a(x) = x`).
    use_bias: Boolean, whether the layer uses a bias vector.
    kernel_initializer: Initializer for the `kernel` weights matrix.
    bias_initializer: Initializer for the bias vector.
    kernel_regularizer: Regularizer function applied to
      the `kernel` weights matrix.
    bias_regularizer: Regularizer function applied to the bias vector.
    activity_regularizer: Regularizer function applied to
      the output of the layer (its "activation").
    kernel_constraint: Constraint function applied to
      the `kernel` weights matrix.
    bias_constraint: Constraint function applied to the bias vector.
  Input shape:
    N-D tensor with shape: `(batch_size, ..., input_dim)`.
    The most common situation would be
    a 2D input with shape `(batch_size, input_dim)`.
  Output shape:
    N-D tensor with shape: `(batch_size, ..., units)`.
    For instance, for a 2D input with shape `(batch_size, input_dim)`,
    the output would have shape `(batch_size, units)`.
  """

  def __init__(self,
               units,
               activation=None,
               use_bias=True,
               kernel_initializer=None,
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               trainable=True,
               weight_norm=True,
               mean_only_batch_norm=True,
               name= None,
               **kwargs):
    super(Dense, self).__init__(trainable=trainable, name=name,activity_regularizer=activity_regularizer,**kwargs)

    self.units = int(units) if not isinstance(units, int) else units
    self.activation = activations.get(activation)
    self.use_bias = use_bias
    self.kernel_initializer = initializers.get(kernel_initializer)
    self.bias_initializer = initializers.get(bias_initializer)
    self.kernel_regularizer = regularizers.get(kernel_regularizer)
    self.bias_regularizer = regularizers.get(bias_regularizer)
    self.kernel_constraint = constraints.get(kernel_constraint)
    self.bias_constraint = constraints.get(bias_constraint)
    self.input_spec = InputSpec(min_ndim=2)
    self.weight_norm = weight_norm,
    self.mean_only_batch_norm = mean_only_batch_norm,
    self.supports_masking = True

  def build(self, input_shape):
    dtype = dtypes.as_dtype(self.dtype or K.floatx())
    if not (dtype.is_floating or dtype.is_complex):
      raise TypeError('Unable to build `Dense` layer with non-floating point '
                      'dtype %s' % (dtype,))

    input_shape = tensor_shape.TensorShape(input_shape)
    last_dim = tensor_shape.dimension_value(input_shape[-1])
    if last_dim is None:
      raise ValueError('The last dimension of the inputs to `Dense` '
                       'should be defined. Found `None`.')
    self.input_spec = InputSpec(min_ndim=2, axes={-1: last_dim})
    self.kernel = self.add_weight('kernel',shape=[last_dim, self.units],initializer=self.kernel_initializer,
    regularizer=self.kernel_regularizer,
    constraint=self.kernel_constraint,
    dtype=self.dtype,
    trainable=True)
    if self.weight_norm :
        self.V = self.add_weight(name='V_weight_norm',
                                     shape=[last_dim, self.units],
                                     dtype=tf.float32,
                                     initializer=tf.random_normal_initializer(0, 0.05),
                                     trainable=True )
        self.g = self.add_weight(name='g_weight_norm',
                                     shape=(self.units,),
                                     initializer=init_ops.ones_initializer(),
                                     dtype=self.dtype,
                                     trainable=True )

    if self.mean_only_batch_norm :
        self.batch_norm_running_average = []
    if self.use_bias:
      self.bias = self.add_weight(
          'bias',
          shape=[self.units,],
          initializer=self.bias_initializer,
          regularizer=self.bias_regularizer,
          constraint=self.bias_constraint,
          dtype=self.dtype,
          trainable=True)
    else:
      self.bias = None
    self.built = True

  def call(self, inputs, training= True):
      inputs = tf.convert_to_tensor ( inputs, dtype=self.dtype )
      shape = inputs.get_shape ().as_list ()

      if self.weight_norm :
          inputs = tf.matmul( inputs, self.V )
          scaler = self.g / tf.math.sqrt ( tf.math.reduce_sum ( tf.math.square ( self.V ), [0] ) )
          outputs = tf.reshape(scaler, [1, self.units] ) * inputs
      else :
          if len ( shape ) > 2 :
              # Broadcasting is required for the inputs.
              outputs = standard_ops.tensordot(inputs, self.kernel, [[len ( shape ) - 1],
                                                                       [0]] )
              # Reshape the output back to the original ndim of the input.
              if not context.executing_eagerly () :
                  output_shape = shape[:-1] + [self.units]
                  outputs.set_shape ( output_shape )
          else :
              outputs = gen_math_ops.mat_mul ( inputs, self.kernel )

      if self.mean_only_batch_norm :
          mean = tf.math.reduce_mean ( outputs)
          if training :
              # If first iteration
              if self.batch_norm_running_average == [] :
                  self.batch_norm_running_average = mean
              else :
                  self.batch_norm_running_average = (self.batch_norm_running_average + mean) / 2
                  outputs = outputs - mean
          else :
              outputs = outputs - self.batch_norm_running_average

      if self.use_bias :
          outputs = nn.bias_add ( outputs, self.bias )
      if self.activation is not None :
          return self.activation ( outputs )  # pylint: disable=not-callable
      return outputs


  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    return input_shape[:-1].concatenate(self.units)

  def get_config(self):
    config = super(Dense, self).get_config()
    config.update({
        'units':
            self.units,
        'activation':
            activations.serialize(self.activation),
        'use_bias':
            self.use_bias,
        'kernel_initializer':
            initializers.serialize(self.kernel_initializer),
        'bias_initializer':
            initializers.serialize(self.bias_initializer),
        'kernel_regularizer':
            regularizers.serialize(self.kernel_regularizer),
        'bias_regularizer':
            regularizers.serialize(self.bias_regularizer),
        'activity_regularizer':
            regularizers.serialize(self.activity_regularizer),
        'kernel_constraint':
            constraints.serialize(self.kernel_constraint),
        'bias_constraint':
            constraints.serialize(self.bias_constraint)
    })
    return config


  def dense(
          inputs, units,
          activation=None,
          use_bias=True,
          kernel_initializer=None,
          bias_initializer=init_ops.zeros_initializer (),
          kernel_regularizer=None,
          bias_regularizer=None,
          activity_regularizer=None,
          kernel_constraint=None,
          bias_constraint=None,
          weight_norm=True,
          mean_only_batch_norm=True,
          trainable=True,
          name=None,
          reuse=None) :
      """Functional interface for the densely-connected layer.

      This layer implements the operation:
      `outputs = activation(inputs.kernel + bias)`
      Where `activation` is the activation function passed as the `activation`
      argument (if not `None`), `kernel` is a weights matrix created by the layer,
      and `bias` is a bias vector created by the layer
      (only if `use_bias` is `True`).

      Arguments:
        inputs: Tensor input.
        units: Integer or Long, dimensionality of the output space.
        activation: Activation function (callable). Set it to None to maintain a
          linear activation.
        use_bias: Boolean, whether the layer uses a bias.
        kernel_initializer: Initializer function for the weight matrix.
          If `None` (default), weights are initialized using the default
          initializer used by `tf.get_variable`.
        bias_initializer: Initializer function for the bias.
        kernel_regularizer: Regularizer function for the weight matrix.
        bias_regularizer: Regularizer function for the bias.
        activity_regularizer: Regularizer function for the output.
        kernel_constraint: An optional projection function to be applied to the
            kernel after being updated by an `Optimizer` (e.g. used to implement
            norm constraints or value constraints for layer weights). The function
            must take as input the unprojected variable and must return the
            projected variable (which must have the same shape). Constraints are
            not safe to use when doing asynchronous distributed training.
        bias_constraint: An optional projection function to be applied to the
            bias after being updated by an `Optimizer`.
        trainable: Boolean, if `True` also add variables to the graph collection
          `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
        name: String, the name of the layer.
        reuse: Boolean, whether to reuse the weights of a previous layer
          by the same name.

      Returns:
        Output tensor the same shape as `inputs` except the last dimension is of
        size `units`.

      Raises:
        ValueError: if eager execution is enabled.
      """
      layer = Dense ( units,
                      activation=activation,
                      use_bias=use_bias,
                      kernel_initializer=kernel_initializer,
                      bias_initializer=bias_initializer,
                      kernel_regularizer=kernel_regularizer,
                      bias_regularizer=bias_regularizer,
                      activity_regularizer=activity_regularizer,
                      kernel_constraint=kernel_constraint,
                      bias_constraint=bias_constraint,
                      trainable=trainable,
                      weight_norm=weight_norm,
                      mean_only_batch_norm=mean_only_batch_norm,
                      name=name,
                      dtype=inputs.dtype.base_dtype,
                      _scope=name,
                      _reuse=reuse )
      return layer.apply(inputs)