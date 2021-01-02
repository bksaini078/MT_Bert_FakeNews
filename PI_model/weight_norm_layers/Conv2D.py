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
"""Keras convolution layers and image transformation layers.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import six
import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import activations
from tensorflow.python.keras import backend
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.engine.input_spec import InputSpec
# imports for backwards namespace compatibility
# pylint: disable=unused-import
from tensorflow.python.keras.layers.pooling import AveragePooling1D
from tensorflow.python.keras.layers.pooling import AveragePooling2D
from tensorflow.python.keras.layers.pooling import AveragePooling3D
from tensorflow.python.keras.layers.pooling import MaxPooling1D
from tensorflow.python.keras.layers.pooling import MaxPooling2D
from tensorflow.python.keras.layers.pooling import MaxPooling3D
# pylint: enable=unused-import
from tensorflow.python.keras.utils import conv_utils
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops, init_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import nn_ops
from tensorflow.python.util.tf_export import keras_export


# pylint: disable=g-classes-have-attributes


class _Conv ( Layer ) :
    """Abstract N-D convolution layer (private, used as implementation base).
    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    Note: layer attributes cannot be modified after the layer has been called
    once (except the `trainable` attribute).
    Arguments:
      rank: An integer, the rank of the convolution, e.g. "2" for 2D convolution.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of n integers, specifying the
        length of the convolution window.
      strides: An integer or tuple/list of n integers,
        specifying the stride length of the convolution.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"`,  `"same"`, or `"causal"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch_size, ..., channels)` while `channels_first` corresponds to
        inputs with shape `(batch_size, channels, ...)`.
      dilation_rate: An integer or tuple/list of n integers, specifying
        the dilation rate to use for dilated convolution.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any `strides` value != 1.
      groups: A positive integer specifying the number of groups in which the
        input is split along the channel axis. Each group is convolved
        separately with `filters / groups` filters. The output is the
        concatenation of all the `groups` results along the channel axis.
        Input channels and `filters` must both be divisible by `groups`.
      activation: Activation function to use.
        If you don't specify anything, no activation is applied.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` the weights of this layer will be marked as
        trainable (and listed in `layer.trainable_weights`).
      name: A string, the name of the layer.
    """

    def __init__(self,
                 rank,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 data_format=None,
                 dilation_rate=1,
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 weight_norm=True,
                 mean_only_batch_norm=True,
                 name=None,
                 conv_op=None,
                 **kwargs) :
        super ( _Conv, self ).__init__ ( trainable=trainable,
                                         name=name, activity_regularizer=regularizers.get ( activity_regularizer ),
                                         **kwargs )
        self.rank = rank

        if isinstance ( filters, float ) :
            filters = int ( filters )
        self.filters = filters
        self.groups = groups or 1
        self.kernel_size = conv_utils.normalize_tuple (
            kernel_size, rank, 'kernel_size' )
        self.strides = conv_utils.normalize_tuple ( strides, rank, 'strides' )
        self.padding = conv_utils.normalize_padding ( padding )
        self.data_format = conv_utils.normalize_data_format ( data_format )
        self.dilation_rate = conv_utils.normalize_tuple (
            dilation_rate, rank, 'dilation_rate' )

        self.activation = activations.get ( activation )
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get ( kernel_initializer )
        self.bias_initializer = initializers.get ( bias_initializer )
        self.kernel_regularizer = regularizers.get ( kernel_regularizer )
        self.bias_regularizer = regularizers.get ( bias_regularizer )
        self.kernel_constraint = constraints.get ( kernel_constraint )
        self.bias_constraint = constraints.get ( bias_constraint )
        self.input_spec = InputSpec ( min_ndim=self.rank + 2 )
        self.weight_norm = weight_norm
        self.mean_only_batch_norm = mean_only_batch_norm

        # self._validate_init ()
        self._is_causal = self.padding == 'causal'
        # self._channels_first = self.data_format == 'channels_first'
        self._tf_data_format = conv_utils.convert_data_format (
            self.data_format, self.rank + 2 )

    # def _validate_init(self) :
    #     if self.filters is not None and self.filters % self.groups != 0 :
    #         raise ValueError (
    #             'The number of filters must be evenly divisible by the number of '
    #             'groups. Received: groups={}, filters={}'.format (
    #                 self.groups, self.filters ) )
    #
    #     if not all ( self.kernel_size ) :
    #         raise ValueError ( 'The argument `kernel_size` cannot contain 0(s). '
    #                            'Received: %s' % (self.kernel_size,) )
    #
    #     if (self.padding == 'causal' and not isinstance ( self,
    #                                                       (Conv1D, SeparableConv1D) )) :
    #         raise ValueError ( 'Causal padding is only supported for `Conv1D`'
    #                            'and `SeparableConv1D`.' )

    def build(self, input_shape) :
        input_shape = tensor_shape.TensorShape ( input_shape )
        input_channel = self._get_input_channel ( input_shape )
        if input_channel % self.groups != 0 :
            raise ValueError (
                'The number of input channels must be evenly divisible by the number '
                'of groups. Received groups={}, but the input has {} channels '
                '(full input shape is {}).'.format ( self.groups, input_channel,
                                                     input_shape ) )
        kernel_shape = self.kernel_size + (input_channel // self.groups,
                                           self.filters)

        self.kernel = self.add_weight (
            name='kernel',
            shape=kernel_shape,
            initializer=self.kernel_initializer,
            regularizer=self.kernel_regularizer,
            constraint=self.kernel_constraint,
            trainable=True,
            dtype=self.dtype )
        if self.weight_norm :
            self.V = self.add_weight ( name='V_weight_norm',
                                         shape=kernel_shape,
                                         dtype=tf.float32,
                                         initializer=tf.random_normal_initializer (
                                             0, 0.05 ),
                                         trainable=True )
            self.g = self.add_weight ( name='g_weight_norm',
                                         shape=(self.filters,),
                                         initializer=init_ops.ones_initializer (),
                                         dtype=self.dtype,
                                         trainable=True )
        if self.mean_only_batch_norm :
            self.batch_norm_running_average = []

        if self.use_bias :
            self.bias = self.add_weight (
                name='bias',
                shape=(self.filters,),
                initializer=self.bias_initializer,
                regularizer=self.bias_regularizer,
                constraint=self.bias_constraint,
                trainable=True,
                dtype=self.dtype )
        else :
            self.bias = None
        channel_axis = self._get_channel_axis ()
        self.input_spec = InputSpec ( min_ndim=self.rank + 2,
                                      axes={channel_axis : input_channel} )

        # Convert Keras formats to TF native formats.
        if self.padding == 'causal' :
            tf_padding = 'VALID'  # Causal padding handled in `call`.
        elif isinstance ( self.padding, six.string_types ) :
            tf_padding = self.padding.upper ()
        else :
            tf_padding = self.padding
        tf_dilations = list ( self.dilation_rate )
        tf_strides = list ( self.strides )

        tf_op_name = self.__class__.__name__
        if tf_op_name == 'Conv1D' :
            tf_op_name = 'conv1d'  # Backwards compat.

        self._convolution_op = functools.partial (
            nn_ops.convolution_v2,
            strides=tf_strides,
            padding=tf_padding,
            dilations=tf_dilations,
            data_format=self._tf_data_format,
            name=tf_op_name )
        self.built = True

    def call(self, inputs, training=True) :
        if self.weight_norm :
            self.kernel = tf.reshape ( self.g, [1, 1, 1, self.filters] ) * tf.nn.l2_normalize ( self.V, [0, 1, 2] )
        if self._is_causal :  # Apply causal padding to inputs for Conv1D.
            inputs = array_ops.pad ( inputs, self._compute_causal_padding ( inputs ) )

        outputs = self._convolution_op(inputs, self.kernel)
        if self.mean_only_batch_norm :
            mean = tf.math.reduce_mean( outputs )
            if training :
                # If first iteration
                if self.batch_norm_running_average == [] :
                    self.batch_norm_running_average = mean
                else :
                    self.batch_norm_running_average = (
                                                              self.batch_norm_running_average + mean) / 2
                outputs = outputs - mean
            else :
                outputs = outputs - self.batch_norm_running_average

        if self.use_bias :
            output_rank = outputs.shape.rank
            if self.rank == 1 and self._channels_first :
                # nn.bias_add does not accept a 1D input tensor.
                bias = array_ops.reshape ( self.bias, (1, self.filters, 1) )
                outputs += bias
            else :
                # Handle multiple batch dimensions.
                if output_rank is not None and output_rank > 2 + self.rank :

                    def _apply_fn(o) :
                        return nn.bias_add ( o, self.bias, data_format=self._tf_data_format )

                    outputs = nn_ops.squeeze_batch_dims (
                        outputs, _apply_fn, inner_rank=self.rank + 1 )
                else :
                    outputs = nn.bias_add (
                        outputs, self.bias, data_format=self._tf_data_format )

        if self.activation is not None :
            return self.activation ( outputs )
        return outputs

    def _spatial_output_shape(self, spatial_input_shape) :
        return [
            conv_utils.conv_output_length (
                length,
                self.kernel_size[i],
                padding=self.padding,
                stride=self.strides[i],
                dilation=self.dilation_rate[i] )
            for i, length in enumerate ( spatial_input_shape )
        ]

    def compute_output_shape(self, input_shape) :
        input_shape = tensor_shape.TensorShape ( input_shape ).as_list ()
        batch_rank = len ( input_shape ) - self.rank - 1
        if self.data_format == 'channels_last' :
            return tensor_shape.TensorShape (
                input_shape[:batch_rank]
                + self._spatial_output_shape ( input_shape[batch_rank :-1] )
                + [self.filters] )
        else :
            return tensor_shape.TensorShape (
                input_shape[:batch_rank] + [self.filters] +
                self._spatial_output_shape ( input_shape[batch_rank + 1 :] ) )

    def _recreate_conv_op(self, inputs) :  # pylint: disable=unused-argument
        return False

    def get_config(self) :
        config = {
            'filters' :
                self.filters,
            'kernel_size' :
                self.kernel_size,
            'strides' :
                self.strides,
            'padding' :
                self.padding,
            'data_format' :
                self.data_format,
            'dilation_rate' :
                self.dilation_rate,
            'groups' :
                self.groups,
            'activation' :
                activations.serialize ( self.activation ),
            'use_bias' :
                self.use_bias,
            'kernel_initializer' :
                initializers.serialize ( self.kernel_initializer ),
            'bias_initializer' :
                initializers.serialize ( self.bias_initializer ),
            'kernel_regularizer' :
                regularizers.serialize ( self.kernel_regularizer ),
            'bias_regularizer' :
                regularizers.serialize ( self.bias_regularizer ),
            'activity_regularizer' :
                regularizers.serialize ( self.activity_regularizer ),
            'kernel_constraint' :
                constraints.serialize ( self.kernel_constraint ),
            'bias_constraint' :
                constraints.serialize ( self.bias_constraint )
        }
        base_config = super ( _Conv, self ).get_config ()
        return dict ( list ( base_config.items () ) + list ( config.items () ) )

    def _compute_causal_padding(self, inputs) :
        """Calculates padding for 'causal' option for 1-d conv layers."""
        left_pad = self.dilation_rate[0] * (self.kernel_size[0] - 1)
        if getattr ( inputs.shape, 'ndims', None ) is None :
            batch_rank = 1
        else :
            batch_rank = len ( inputs.shape ) - 2
        if self.data_format == 'channels_last' :
            causal_padding = [[0, 0]] * batch_rank + [[left_pad, 0], [0, 0]]
        else :
            causal_padding = [[0, 0]] * batch_rank + [[0, 0], [left_pad, 0]]
        return causal_padding

    def _get_channel_axis(self) :
        if self.data_format == 'channels_first' :
            return -1 - self.rank
        else :
            return -1

    def _get_input_channel(self, input_shape) :
        channel_axis = self._get_channel_axis ()
        if input_shape.dims[channel_axis].value is None :
            raise ValueError ( 'The channel dimension of the inputs '
                               'should be defined. Found `None`.' )
        return int ( input_shape[channel_axis] )

    def _get_padding_op(self) :
        if self.padding == 'causal' :
            op_padding = 'valid'
        else :
            op_padding = self.padding
        if not isinstance ( op_padding, (list, tuple) ) :
            op_padding = op_padding.upper ()
        return op_padding


@keras_export ( 'keras.layers.Conv2D', 'keras.layers.Convolution2D' )
class Conv2D ( _Conv ) :
    """2D convolution layer (e.g. spatial convolution over images).
    This layer creates a convolution kernel that is convolved
    with the layer input to produce a tensor of
    outputs. If `use_bias` is True,
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.
    When using this layer as the first layer in a model,
    provide the keyword argument `input_shape`
    (tuple of integers, does not include the sample axis),
    e.g. `input_shape=(128, 128, 3)` for 128x128 RGB pictures
    in `data_format="channels_last"`.
    Examples:
    >>> # The inputs are 28x28 RGB images with `channels_last` and the batch
    >>> # size is 4.
    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv2D(
    ... 2, 3, activation='relu', input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 26, 26, 2)
    >>> # With `dilation_rate` as 2.
    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv2D(
    ... 2, 3, activation='relu', dilation_rate=2, input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 7, 26, 26, 2)
    >>> # With `padding` as "same".
    >>> input_shape = (4, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv2D(
    ... 2, 3, activation='relu', padding="same", input_shape=input_shape[1:])(x)
    >>> print(y.shape)
    (4, 28, 28, 2)
    >>> # With extended batch shape [4, 7]:
    >>> input_shape = (4, 7, 28, 28, 3)
    >>> x = tf.random.normal(input_shape)
    >>> y = tf.keras.layers.Conv2D(
    ... 2, 3, activation='relu', input_shape=input_shape[2:])(x)
    >>> print(y.shape)
    (4, 7, 26, 26, 2)
    Arguments:
      filters: Integer, the dimensionality of the output space (i.e. the number of
        output filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the height
        and width of the 2D convolution window. Can be a single integer to specify
        the same value for all spatial dimensions.
      strides: An integer or tuple/list of 2 integers, specifying the strides of
        the convolution along the height and width. Can be a single integer to
        specify the same value for all spatial dimensions. Specifying any stride
        value != 1 is incompatible with specifying any `dilation_rate` value != 1.
      padding: one of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs. `channels_last` corresponds
        to inputs with shape `(batch_size, height, width, channels)` while
        `channels_first` corresponds to inputs with shape `(batch_size, channels,
        height, width)`. It defaults to the `image_data_format` value found in
        your Keras config file at `~/.keras/keras.json`. If you never set it, then
        it will be `channels_last`.
      dilation_rate: an integer or tuple/list of 2 integers, specifying the
        dilation rate to use for dilated convolution. Can be a single integer to
        specify the same value for all spatial dimensions. Currently, specifying
        any `dilation_rate` value != 1 is incompatible with specifying any stride
        value != 1.
      groups: A positive integer specifying the number of groups in which the
        input is split along the channel axis. Each group is convolved separately
        with `filters / groups` filters. The output is the concatenation of all
        the `groups` results along the channel axis. Input channels and `filters`
        must both be divisible by `groups`.
      activation: Activation function to use. If you don't specify anything, no
        activation is applied (see `keras.activations`).
      use_bias: Boolean, whether the layer uses a bias vector.
      kernel_initializer: Initializer for the `kernel` weights matrix (see
        `keras.initializers`).
      bias_initializer: Initializer for the bias vector (see
        `keras.initializers`).
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix (see `keras.regularizers`).
      bias_regularizer: Regularizer function applied to the bias vector (see
        `keras.regularizers`).
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation") (see `keras.regularizers`).
      kernel_constraint: Constraint function applied to the kernel matrix (see
        `keras.constraints`).
      bias_constraint: Constraint function applied to the bias vector (see
        `keras.constraints`).
    Input shape:
      4+D tensor with shape: `batch_shape + (channels, rows, cols)` if
        `data_format='channels_first'`
      or 4+D tensor with shape: `batch_shape + (rows, cols, channels)` if
        `data_format='channels_last'`.
    Output shape:
      4+D tensor with shape: `batch_shape + (filters, new_rows, new_cols)` if
      `data_format='channels_first'` or 4+D tensor with shape: `batch_shape +
        (new_rows, new_cols, filters)` if `data_format='channels_last'`.  `rows`
        and `cols` values might have changed due to padding.
    Returns:
      A tensor of rank 4+ representing
      `activation(conv2d(inputs, kernel) + bias)`.
    Raises:
      ValueError: if `padding` is `"causal"`.
      ValueError: when both `strides > 1` and `dilation_rate > 1`.
    """

    def __init__(self,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 data_format=None,
                 dilation_rate=(1, 1),
                 groups=1,
                 activation=None,
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 trainable=True,
                 weight_norm=True,
                 mean_only_batch_norm=True,
                 **kwargs) :
        super ( Conv2D, self ).__init__ (
            rank=2,
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            groups=groups,
            activation=activations.get ( activation ),
            use_bias=use_bias,
            kernel_initializer=initializers.get ( kernel_initializer ),
            bias_initializer=initializers.get ( bias_initializer ),
            kernel_regularizer=regularizers.get ( kernel_regularizer ),
            bias_regularizer=regularizers.get ( bias_regularizer ),
            activity_regularizer=regularizers.get ( activity_regularizer ),
            kernel_constraint=constraints.get ( kernel_constraint ),
            bias_constraint=constraints.get ( bias_constraint ),
            trainable=trainable,
            weight_norm=weight_norm,
            mean_only_batch_norm=mean_only_batch_norm,
            **kwargs )


def conv2d(inputs, filters, kernel_size,
           strides=(1, 1),
           padding='valid',
           data_format='channels_last',
           dilation_rate=(1, 1),
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
    """Functional interface for the 2D convolution layer.
    This layer creates a convolution kernel that is convolved
    (actually cross-correlated) with the layer input to produce a tensor of
    outputs. If `use_bias` is True (and a `bias_initializer` is provided),
    a bias vector is created and added to the outputs. Finally, if
    `activation` is not `None`, it is applied to the outputs as well.

    Arguments:
      inputs: Tensor input.
      filters: Integer, the dimensionality of the output space (i.e. the number
        of filters in the convolution).
      kernel_size: An integer or tuple/list of 2 integers, specifying the
        height and width of the 2D convolution window.
        Can be a single integer to specify the same value for
        all spatial dimensions.
      strides: An integer or tuple/list of 2 integers,
        specifying the strides of the convolution along the height and width.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Specifying any stride value != 1 is incompatible with specifying
        any `dilation_rate` value != 1.
      padding: One of `"valid"` or `"same"` (case-insensitive).
      data_format: A string, one of `channels_last` (default) or `channels_first`.
        The ordering of the dimensions in the inputs.
        `channels_last` corresponds to inputs with shape
        `(batch, height, width, channels)` while `channels_first` corresponds to
        inputs with shape `(batch, channels, height, width)`.

      dilation_rate: An integer or tuple/list of 2 integers, specifying
        the dilation rate to use for dilated convolution.
        Can be a single integer to specify the same value for
        all spatial dimensions.
        Currently, specifying any `dilation_rate` value != 1 is
        incompatible with specifying any stride value != 1.
      activation: Activation function. Set it to None to maintain a
        linear activation.
      use_bias: Boolean, whether the layer uses a bias.
      kernel_initializer: An initializer for the convolution kernel.
      bias_initializer: An initializer for the bias vector. If None, the default
        initializer will be used.
      kernel_regularizer: Optional regularizer for the convolution kernel.
      bias_regularizer: Optional regularizer for the bias vector.
      activity_regularizer: Optional regularizer function for the output.
      kernel_constraint: Optional projection function to be applied to the
          kernel after being updated by an `Optimizer` (e.g. used to implement
          norm constraints or value constraints for layer weights). The function
          must take as input the unprojected variable and must return the
          projected variable (which must have the same shape). Constraints are
          not safe to use when doing asynchronous distributed training.
      bias_constraint: Optional projection function to be applied to the
          bias after being updated by an `Optimizer`.
      trainable: Boolean, if `True` also add variables to the graph collection
        `GraphKeys.TRAINABLE_VARIABLES` (see `tf.Variable`).
      name: A string, the name of the layer.
      reuse: Boolean, whether to reuse the weights of a previous layer
        by the same name.

    Returns:
      Output tensor.

    Raises:
      ValueError: if eager execution is enabled.
"""
    layer = Conv2D (
        filters=filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        data_format=data_format,
        dilation_rate=dilation_rate,
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
        name=name,
        dtype=inputs.dtype.base_dtype,
        _reuse=reuse,
        _scope=name )
    return layer.apply ( inputs )
