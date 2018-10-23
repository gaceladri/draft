import tensorflow as tf
import numpy as np

import six
import functools
import math

from tensor2tensor import common_attention, common_layers

DEFAULT_DEV_STRING = "existing_device"


def transformer_prepare_encoder(inputs, hparams, features=None):
    """Prepare one shard of the model for the encoder.

    Args:
      inputs: a Tensor.
      hparams: run hyperparameters
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      encoder_input: a Tensor, bottom of encoder stack
      encoder_self_attention_bias: a bias tensor for use in encoder self-attention
      encoder_decoder_attention_bias: a bias tensor for use in encoder-decoder
        attention
    """
    ishape_static = inputs.shape.as_list()
    encoder_input = inputs

    # Usual case - not a packed dataset.
    encoder_padding = common_attention.embedding_to_padding(encoder_input)
    ignore_padding = common_attention.attention_bias_ignore_padding(
        encoder_padding)
    encoder_self_attention_bias = ignore_padding
    encoder_decoder_attention_bias = ignore_padding
    inputs_position = None
    if hparams.proximity_bias:
        encoder_self_attention_bias += common_attention.attention_bias_proximal(
            common_layers.shape_list(inputs)[1])

    if inputs_position is not None:
        encoder_input = common_attention.add_timing_signal_1d_given_position(
            encoder_input, inputs_position)
    else:
        encoder_input = common_attention.add_timing_signal_1d(
            encoder_input)
    return (encoder_input, encoder_self_attention_bias,
            encoder_decoder_attention_bias)


def transformer_prepare_decoder(targets, hparams, features=None):
    """Prepare one shard of the model for the decoder.

    Args:
      targets: a Tensor.
      hparams: run hyperparameters
      features: optionally pass the entire features dictionary as well.
        This is needed now for "packed" datasets.

    Returns:
      decoder_input: a Tensor, bottom of decoder stack
      decoder_self_attention_bias: a bias tensor for use in decoder self-attention
    """
    if hparams.causal_decoder_self_attention:
        # Causal attention.
        if hparams.prepend_mode == "prepend_inputs_full_attention":
            decoder_self_attention_bias = (
                common_attention.attention_bias_prepend_inputs_full_attention(
                    common_attention.embedding_to_padding(targets)))
        else:
            decoder_self_attention_bias = (
                common_attention.attention_bias_lower_triangle(
                    common_layers.shape_list(targets)[1]))
    else:
        # Full attention.
        decoder_padding = common_attention.embedding_to_padding(targets)
        decoder_self_attention_bias = (
            common_attention.attention_bias_ignore_padding(decoder_padding))

    if hparams.proximity_bias:
        decoder_self_attention_bias += common_attention.attention_bias_proximal(
            common_layers.shape_list(targets)[1])
    decoder_input = common_layers.shift_right_3d(targets)
    decoder_input = common_attention.add_timing_signal_1d(
        decoder_input)

    return (decoder_input, decoder_self_attention_bias)


def layer_norm_vars(filters):
    """Create Variables for layer norm."""
    scale = tf.get_variable(
        "layer_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "layer_norm_bias", [filters], initializer=tf.zeros_initializer())
    return scale, bias


def layer_norm_compute(x, epsilon, scale, bias):
    """Layer norm raw computation."""
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
    mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
    variance = tf.reduce_mean(tf.square(x - mean), axis=[-1], keepdims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def layer_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalize the tensor x, averaging over the last dimension."""
    if filters is None:
        filters = shape_list(x)[-1]
    with tf.variable_scope(
            name, default_name="layer_norm", values=[x], reuse=reuse):
        scale, bias = layer_norm_vars(filters)
        return layer_norm_compute(x, epsilon, scale, bias)


def group_norm(x, filters=None, num_groups=8, epsilon=1e-5):
    """Group normalization as in https://arxiv.org/abs/1803.08494."""
    x_shape = shape_list(x)
    if filters is None:
        filters = x_shape[-1]
    assert len(x_shape) == 4
    assert filters % num_groups == 0
    # Prepare variables.
    scale = tf.get_variable(
        "group_norm_scale", [filters], initializer=tf.ones_initializer())
    bias = tf.get_variable(
        "group_norm_bias", [filters], initializer=tf.zeros_initializer())
    epsilon, scale, bias = [cast_like(t, x) for t in [epsilon, scale, bias]]
    # Reshape and compute group norm.
    x = tf.reshape(x, x_shape[:-1] + [num_groups, filters // num_groups])
    # Calculate mean and variance on heights, width, channels (not groups).
    mean, variance = tf.nn.moments(x, [1, 2, 4], keep_dims=True)
    norm_x = (x - mean) * tf.rsqrt(variance + epsilon)
    return tf.reshape(norm_x, x_shape) * scale + bias


def noam_norm(x, epsilon=1.0, name=None):
    """One version of layer normalization."""
    with tf.name_scope(name, default_name="noam_norm", values=[x]):
        shape = x.get_shape()
        ndims = len(shape)
        return (tf.nn.l2_normalize(x, ndims - 1, epsilon=epsilon) * tf.sqrt(
            tf.to_float(shape[-1])))


def l2_norm(x, filters=None, epsilon=1e-6, name=None, reuse=None):
    """Layer normalization with l2 norm."""
    if filters is None:
        filters = shape_list(x)[-1]
    with tf.variable_scope(name, default_name="l2_norm", values=[x], reuse=reuse):
        scale = tf.get_variable(
            "l2_norm_scale", [filters], initializer=tf.ones_initializer())
        bias = tf.get_variable(
            "l2_norm_bias", [filters], initializer=tf.zeros_initializer())
        epsilon, scale, bias = [cast_like(t, x)
                                for t in [epsilon, scale, bias]]
        mean = tf.reduce_mean(x, axis=[-1], keepdims=True)
        l2norm = tf.reduce_sum(tf.square(x - mean), axis=[-1], keepdims=True)
        norm_x = (x - mean) * tf.rsqrt(l2norm + epsilon)
        return norm_x * scale + bias


def apply_norm(x, norm_type, depth, epsilon):
    """Apply Normalization."""
    if norm_type == "layer":
        return layer_norm(x, filters=depth, epsilon=epsilon)
    if norm_type == "group":
        return group_norm(x, filters=depth, epsilon=epsilon)
    if norm_type == "batch":
        return tf.layers.batch_normalization(x, epsilon=epsilon)
    if norm_type == "noam":
        return noam_norm(x, epsilon)
    if norm_type == "l2":
        return l2_norm(x, filters=depth, epsilon=epsilon)
    if norm_type == "none":
        return x
    raise ValueError("Parameter normalizer_fn must be one of: 'layer', 'batch',"
                     "'noam', 'lr', 'none'.")


def zero_add(previous_value, x, name=None, reuse=None):
    """Resnet connection with zero initialization.

    Another type of resnet connection which returns previous_value + gamma * x.
    gamma is a trainable scalar and initialized with zero. It is useful when a
    module is plugged into a trained model and we want to make sure it matches the
    original model's performance.

    Args:
      previous_value:  A tensor.
      x: A tensor.
      name: name of variable scope; defaults to zero_add.
      reuse: reuse scope.

    Returns:
      previous_value + gamma * x.
    """
    with tf.variable_scope(name, default_name="zero_add", reuse=reuse):
        gamma = tf.get_variable(
            "gamma", (), initializer=tf.zeros_initializer())
        return previous_value + gamma * x


def layer_prepostprocess(previous_value,
                         x,
                         sequence,
                         dropout_rate,
                         norm_type,
                         depth,
                         epsilon,
                         default_name,
                         name=None,
                         dropout_broadcast_dims=None):
    """Apply a sequence of functions to the input or output of a layer.

    The sequence is specified as a string which may contain the following
    characters:
      a: add previous_value
      n: apply normalization
      d: apply dropout
      z: zero add

    For example, if sequence=="dna", then the output is
      previous_value + normalize(dropout(x))

    Args:
      previous_value: A Tensor, to be added as a residual connection ('a')
      x: A Tensor to be transformed.
      sequence: a string.
      dropout_rate: a float
      norm_type: a string (see apply_norm())
      depth: an integer (size of last dimension of x).
      epsilon: a float (parameter for normalization)
      default_name: a string
      name: a string
      dropout_broadcast_dims:  an optional list of integers less than 3
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.

    Returns:
      a Tensor
    """
    with tf.variable_scope(name, default_name=default_name):
        if sequence == "none":
            return x
        for c in sequence:
            if c == "a":
                x += previous_value
            elif c == "z":
                x = zero_add(previous_value, x)
            elif c == "n":
                x = apply_norm(x, norm_type, depth, epsilon)
            else:
                assert c == "d", ("Unknown sequence step %s" % c)
                x = dropout_with_broadcast_dims(
                    x, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        return x


def comma_separated_string_to_integer_list(s):
    return [int(i) for i in s.split(",") if i]


def layer_preprocess(layer_input, hparams):
    """Apply layer preprocessing.

    See layer_prepostprocess() for details.

    A hyperparameters object is passed for convenience.  The hyperparameters
    that may be used are:

      layer_preprocess_sequence
      layer_prepostprocess_dropout
      norm_type
      hidden_size
      norm_epsilon

    Args:
      layer_input: a Tensor
      hparams: a hyperparameters object.

    Returns:
      a Tensor
    """
    assert "a" not in hparams.layer_preprocess_sequence, (
        "No residual connections allowed in hparams.layer_preprocess_sequence")
    assert "z" not in hparams.layer_preprocess_sequence, (
        "No residual connections allowed in hparams.layer_preprocess_sequence")
    return layer_prepostprocess(
        None,
        layer_input,
        sequence=hparams.layer_preprocess_sequence,
        dropout_rate=hparams.layer_prepostprocess_dropout,
        norm_type=hparams.norm_type,
        depth=None,
        epsilon=hparams.norm_epsilon,
        dropout_broadcast_dims=comma_separated_string_to_integer_list(
            getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
        default_name="layer_prepostprocess")


def layer_postprocess(layer_input, layer_output, hparams):
    """Apply layer postprocessing.

    See layer_prepostprocess() for details.

    A hyperparameters object is passed for convenience.  The hyperparameters
    that may be used are:

      layer_postprocess_sequence
      layer_prepostprocess_dropout
      norm_type
      hidden_size
      norm_epsilon

    Args:
      layer_input: a Tensor
      layer_output: a Tensor
      hparams: a hyperparameters object.

    Returns:
      a Tensor
    """
    return layer_prepostprocess(
        layer_input,
        layer_output,
        sequence=hparams.layer_postprocess_sequence,
        dropout_rate=hparams.layer_prepostprocess_dropout,
        norm_type=hparams.norm_type,
        depth=None,
        epsilon=hparams.norm_epsilon,
        dropout_broadcast_dims=comma_separated_string_to_integer_list(
            getattr(hparams, "layer_prepostprocess_dropout_broadcast_dims", "")),
        default_name="layer_postprocess")


def _extract_layer_types(hparams):
    SEP_ENCODEC = "#"
    SEP_LAYER = "/"
    SEP_FF = "-"
    """Parse the layer string.

    Returns:
      list[tuple[str, str]]: Encoder layers: list of (attention, feed-forward)
      list[tuple[str, str, str]]: Decoder layers: list of (self-attention,
        enc-dec attention, feed-forward)
    """
    hparams = hparams
    layer_types = hparams.layer_types

    # If the architecture has not explicitly been set, we just construct a
    # standard transformer with the fallback values
    if not layer_types:
        layer_types = SEP_LAYER.join(
            [hparams.default_att] * hparams.num_hidden_layers)

    # If encoder not explicitly defined, the encoder will have the same
    # structure as the decoder
    layer_types = layer_types.split(SEP_ENCODEC)
    if len(layer_types) == 1:
        layer_types *= 2

    # Some models don't need the encoder (ex: language modeling)
    # TODO(epot): What are the other conditions (has_input ?)
    if hparams.prepend_mode != "none":
        layer_types[0] = ""

    # Extend the blocks and fill them with the default values if not specified
    final_layers = ([], [])
    for i, blocks_str_joined in enumerate(layer_types):
        for blocks_str in blocks_str_joined.split(SEP_LAYER):
            if not blocks_str:
                continue
            blocks_list = blocks_str.split(SEP_FF)
            # Eventually use the fallback values for the layer_types. If the
            # encoder is empty, do not use the enco-deco attention.
            self_att = blocks_list[0] or hparams.default_att
            ende_att = hparams.default_att if layer_types[0] else "_"
            ff = hparams.default_ff
            if len(blocks_list) > 1:
                ff = blocks_list[-1]
            if len(blocks_list) == 3:
                ende_att = blocks_list[1]
            if i == 0:  # Encoder
                blocks_tuple = (self_att, ff)
            elif i == 1:  # Decoder
                blocks_tuple = (self_att, ende_att, ff)
            final_layers[i].append(blocks_tuple)

    return final_layers


def conv_internal(conv_fn, inputs, filters, kernel_size, **kwargs):
    """Conditional conv_fn making kernel 1d or 2d depending on inputs shape."""
    static_shape = inputs.get_shape()
    if not static_shape or len(static_shape) != 4:
        raise ValueError("Inputs to conv must have statically known rank 4. "
                         "Shape: " + str(static_shape))
    # Add support for left padding.
    if kwargs.get("padding") == "LEFT":
        dilation_rate = (1, 1)
        if "dilation_rate" in kwargs:
            dilation_rate = kwargs["dilation_rate"]
        assert kernel_size[0] % 2 == 1 and kernel_size[1] % 2 == 1
        height_padding = 2 * (kernel_size[0] // 2) * dilation_rate[0]
        cond_padding = tf.cond(
            tf.equal(shape_list(inputs)[2], 1), lambda: tf.constant(0),
            lambda: tf.constant(2 * (kernel_size[1] // 2) * dilation_rate[1]))
        width_padding = 0 if static_shape[2] == 1 else cond_padding
        padding = [[0, 0], [height_padding, 0], [width_padding, 0], [0, 0]]
        inputs = tf.pad(inputs, padding)
        # Set middle two dimensions to None to prevent convolution from complaining
        inputs.set_shape([static_shape[0], None, None, static_shape[3]])
        kwargs["padding"] = "VALID"

    def conv2d_kernel(kernel_size_arg, name_suffix):
        """Call conv2d but add suffix to name."""
        name = "{}_{}".format(kwargs.get("name", "conv"), name_suffix)
        original_name = kwargs.pop("name", None)
        original_force2d = kwargs.pop("force2d", None)
        result = conv_fn(inputs, filters, kernel_size_arg, name=name, **kwargs)
        if original_name is not None:
            kwargs["name"] = original_name  # Restore for other calls.
        if original_force2d is not None:
            kwargs["force2d"] = original_force2d
        return result

    return conv2d_kernel(kernel_size, "single")


def conv(inputs, filters, kernel_size, dilation_rate=(1, 1), **kwargs):
    return conv_internal(
        tf.layers.conv2d,
        inputs,
        filters,
        kernel_size,
        dilation_rate=dilation_rate,
        **kwargs)


def conv1d(inputs, filters, kernel_size, dilation_rate=1, **kwargs):
    return tf.squeeze(
        conv(
            tf.expand_dims(inputs, 2),
            filters, (kernel_size, 1),
            dilation_rate=(dilation_rate, 1),
            **kwargs), 2)


def add_scope(scope=None, scope_fn=None):
    """Return a decorator which add a TF name/variable scope to a function.

    Note that the function returned by the decorator accept an additional 'name'
    parameter, which can overwrite the name scope given when the function is
    created.

    Args:
        scope (str): name of the scope. If None, the function name is used.
        scope_fn (fct): Either tf.name_scope or tf.variable_scope

    Returns:
        fct: the add_scope decorator
    """
    def decorator(f):

        @functools.wraps(f)
        def decorated(*args, **kwargs):
            # Python 2 hack for keyword only args
            name = kwargs.pop("name", None)
            with scope_fn(name or scope or f.__name__):
                return f(*args, **kwargs)
        return decorated

    return decorator


def add_var_scope(scope=None):
    return add_scope(scope, scope_fn=tf.variable_scope)


def add_name_scope(scope=None):
    return add_scope(scope, scope_fn=tf.name_scope)


@add_name_scope()
def combine_heads(x):
    """Inverse of split_heads.

    Args:
      x: a Tensor with shape [batch, num_heads, length, channels / num_heads]

    Returns:
      a Tensor with shape [batch, length, channels]
    """
    return combine_last_two_dimensions(tf.transpose(x, [0, 2, 1, 3]))


@add_name_scope()
def combine_last_two_dimensions(x):
    """Reshape x so that the last two dimension become one.

    Args:
      x: a Tensor with shape [..., a, b]

    Returns:
      a Tensor with shape [..., ab]
    """
    x_shape = shape_list(x)
    a, b = x_shape[-2:]
    return tf.reshape(x, x_shape[:-2] + [a * b])


@add_name_scope()
def get_timing_signal_1d(length,
                         channels,
                         min_timescale=1.0,
                         max_timescale=1.0e4,
                         start_index=0):
    """Gets a bunch of sinusoids of different frequencies.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    expressed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        length: scalar, length of timing signal sequence.
        channels: scalar, size of timing embeddings to create. The number of
            different timescales is equal to channels / 2.
        min_timescale: a float
        max_timescale: a float
        start_index: index of first position

    Returns:
        a Tensor of timing signals [1, length, channels]
    """
    positions = tf.to_float(tf.range(length) + start_index)
    num_timescales = channels // 2
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) / (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    scaled_time = tf.expand_dims(positions, 1) * \
        tf.expand_dims(inv_timescales, 0)
    signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
    signal = tf.pad(signal, [[0, 0], [0, tf.mod(channels, 2)]])
    signal = tf.shape(signal, [1, length, channels])
    return signal


@add_name_scope()
def add_timing_signal_1d(x, min_timescale=1, max_timescale=1e4, start_index=0):
    """Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a different
    frequency and phase.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(x+y) and cos(x+y) can be
    experessed in terms of y, sin(x) and cos(x).

    In particular, we use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels / 2. For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, length, channels]
        min_timescale: a float
        max_timescale: a float
        start_index: index of first position

    Returns:
        a Tensor the same shape as x.
    """
    length = shape_list(x)[1]
    channels = shape_list(x)[2]
    signal = get_timing_signal_1d(
        length, channels, min_timescale, max_timescale, start_index)
    return x + signal


@add_name_scope()
def remove_pad(x, pad_remover, mode):
    """Remove padding by concatenating all dimension into one.

    Args:
      x (tf.Tensor): input of shape [batch_size, length, depth]
      pad_remover (obj): a PadRemover object
      mode (ModeKeys): infer, train or eval. If inference, the padding remover is
        not applied

    Returns:
      tf.Tensor of shape [1,length_nonpad,depth] where
        length_nonpad <= batch_size*length
    """
    # Concatenate all tokens (without padding)
    x = flatten_all_but_last(x)

    # Remove padding for training and eval
    if mode != ModeKeys.PREDICT:
        # This is a hack to allows inference when the <go> token
        # is detected as padding and removed. This works for now because there is
        # no padding at inference.
        x = pad_remover.remove(x)

    x = tf.expand_dims(x, axis=0)  # Now batch_size=1
    return x


@add_name_scope()
def restore_pad(x, ref_x, pad_remover, mode):
    x = tf.squeeze(x, axis=0)
    if mode != ModeKeys.PREDICT:
        x = pad_remover.restore(x)
    x = common_layers.reshape_like(x, ref_x)
    return x


@add_name_scope("map_ids")
def map_ids(x, indices, map_fn):
    """Apply a function to each coordinate ids of a multidimensional tensor.

    This allows to process each sequence of a batch independently. This is
    similar to tf.map_fn but with tensor where the batch dim has been flatten.

    Warning: The indices ids have to be contiguous and ordered in memory as the
    output vector for each of the ids are simply concatenated after being
    processed.
    Ex: if your indices are [0,2,2,1,2,0], the output will contains the processed
    rows in the following order: [0,0,1,2,2,2]

    Args:
      x (Tensor): The tensor to be dispatched of shape [length,...]
      indices (Tensor): A int32 tensor of size [length, 1] containing the batch
        coordinate of x
      map_fn (fct): Function called for every ids of the original tensor. Take
        as input a tensor of same rank than x and from shape [length_id,...] with
        length_id <= length. Isn't called if length_id == 0

    Returns:
      a tensor of same shape as x, where each elements has been processed
    """
    indices = tf.reshape(indices, [-1])

    t_i = tf.constant(0)
    # batch_coordinates start at 0
    t_batch_size = tf.reduce_max(indices) + 1

    # ta_stack_out will store the intermediate results for each individual id
    # As alternative to tf.TensorArray, scatter_update could potentially be used
    # but that would require an additional mutable tensor.
    ta_stack_out = tf.TensorArray(
        x.dtype,
        size=t_batch_size,
    )

    # Then we iterate over each sequence individually and compute the
    # transformation for each id
    while_condition = lambda t_i, *args: tf.less(t_i, t_batch_size)

    def body(t_i, ta_stack_out):
        """Loop body."""
        # Gather the ids
        current_ids = tf.to_int32(tf.where(tf.equal(indices, t_i)))
        t_row = tf.gather_nd(x, indices=current_ids)

        # TODO(epot): Should not call map_fn if t_row size is 0

        # Apply transformation to each id
        # Restore batch_dim=1 as most function expect [batch_dim, length, ...] as
        # input
        t_row = tf.expand_dims(t_row, axis=0)
        t_row = map_fn(t_row)
        t_row = tf.squeeze(t_row, axis=0)  # Squeeze for concatenation
        ta_stack_out = ta_stack_out.write(t_i, t_row)

        return [tf.add(t_i, 1), ta_stack_out]  # ++i

    # Run the loop, equivalent to:
    # stack_out = []
    # while i < batch_size:
    #   stack_out.expand(map_fn(x[indices==i]))
    _, ta_stack_out = tf.while_loop(while_condition, body, [t_i, ta_stack_out])

    # Merge all results
    return ta_stack_out.concat()


@add_name_scope()
def coordinate_tensor(shape, axis):
    """Return a tensor with given shape containing coordinate along given axis.

    Args:
      shape: a Tensor representing the shape of the output Tensor
      axis: an integer

    Returns:
      A tensor with the shape and type tf.int32, where each elements its
      coordinate along the given axis.
    """
    if axis < 0:
        # Convert to positive for the one_hot indice
        axis = tf.size(shape) + axis

    r = tf.range(shape[axis])
    r_shape = tf.one_hot(
        axis, tf.size(shape), on_value=-1, off_value=1, dtype=tf.int32)
    return tf.zeros(shape, dtype=tf.int32) + tf.reshape(r, r_shape)


def shift_rigth_3d(x, pad_value=None):
    """Shift the second dimension of x right by one."""
    if pad_value is None:
        shifted_targets = tf.pad(x, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]
    else:
        shifted_targets = tf.concat([pad_value, x], axis=1)[:, :-1, :]
    return shifted_targets


def shape_list(x):
    """Return list of dims, statically where possible."""
    x = tf.convert_to_tensor(x)

    # If unknown rank, return dynamic shape
    if x.get_shape().dims is None:
        return tf.shape(x)

    static = x.get_shape().as_list()
    shape = tf.shape(x)

    ret = []
    for i in range(len(static)):
        dim = static[i]
        if dim is None:
            dim = shape[i]
        ret.append(dim)
    return ret


def flatten_all_but_last(a):
    """Flatten all dimensions of a except the last."""
    ret = tf.reshape(a, [-1, tf.shape(a)[-1]])
    if not tf.contrib.eager.in_eager_mode():
        ret.set_shape([None] + a.get_shape().as_list()[-1:])
    return ret


def reshape_like(a, b):
    """Reshapes a to match the shape of b in all but the last dimension."""
    ret = tf.reshape(a, tf.concat([tf.shape(b)[:-1], tf.shape(a)[-1:]], 0))
    if not tf.contrib.eager.in_eager_mode():
        ret.set_shape(b.get_shape().as_list()[
                      :-1] + a.get_shape().as_list()[-1:])
    return ret


def cv_squared(x):
    """The squared coefficient of variation of a sample.

    Useful as a loss to encourage a positive distribution to be more uniform.
    Epsilons added for numerical stability.
    Returns 0 for an empty Tensor.

    Args:
      x: a 'Tensor'.

    Returns:
      a 'Scalar'.
    """
    epsilon = 1e-10
    float_size = tf.to_float(tf.size(x)) + epsilon
    mean = tf.reduce_sum(x) / float_size
    variance = tf.reduce_sum(tf.square(x - mean)) / float_size
    return variance / (tf.square(mean) + epsilon)


from tensorflow.python.framework import function


@function.Defun(
    python_grad_func=lambda x, dy: tf.convert_to_tensor(dy),
    shape_func=lambda op: [op.inputs[0].get_shape()])
def convert_gradient_to_tensor(x):
    """Identity operation whose gradient is converted to a 'Tensor'.

    Currently, the gradient to 'tf.concat' is particularly expensive to
    compute if dy is an 'IndexedSlices' (a lack of GPU implementation
    forces the gradient operation onto CPU). This situation occurs when the output
    of the 'tf.concat' is eventually passed to 'tf.gather'.
    It is sometimes faster to convert the gradient to a 'Tensor', so as
    to get the cheaper gradient for 'tf.concat'. To do this, replace
    'tf.conact(x)' with 'conver_gradient_to_tensor(tf.concat(x))'.

    Args:
      x: A 'Tensor'.

    Returns:
      The input 'Tensor'.
    """
    return x


class SparseDispatcher(object):
    """Helper for implementing a mixture of experts.

    The purpose of this class is to create input minibatches for the
    experts and to combine the results of the experts to form a unified
    output tensor.

    There are two functions:
        dispatch - take an input Tensor and create input Tensors for each expert.
        combine - take output Tensors from each expert and form a combined output
        Tensor.  Outputs from different experts for the same batch element are
        summed together, weighted by the provided "gates".

    The class is initialized with a "gates" Tensor, which specifies which
    batch elements go to which experts, and the weights to use when combining
    the outputs.  Batch element b is sent to expert e iff gates[b, e] != 0.

    The inputs and outputs are all two-dimensional [batch, depth].
    Caller is responsible for collapsing additional dimensions prior to
    calling this class and reshaping the output to the original shape.
    See common_layers.reshape_like().

    Example use:

    gates: a float32 `Tensor` with shape `[batch_size, num_experts]`
    inputs: a float32 `Tensor` with shape `[batch_size, input_size]`
    experts: a list of length `num_experts` containing sub-networks.

        dispatcher = SparseDispatcher(num_experts, gates)
        expert_inputs = dispatcher.dispatch(inputs)
        expert_outputs = [experts[i](expert_inputs[i])
                                     for i in range(num_experts)]
        outputs = dispatcher.combine(expert_outputs)

    The preceding code sets the output for a particular example b to:
    output[b] = Sum_i(gates[b, i] * experts[i](inputs[b]))

    This class takes advantage of sparsity in the gate matrix by including in the
    `Tensor`s for expert i only the batch elements for which `gates[b, i] > 0`.
    """

    def __init__(self, num_experts, gates):
        """Create a SparseDispatcher.

        Args:
          num_experts: an integer.
          gates: a 'Tensor' of shape '[batch_size, num_experts]'.

        Returns:
          a SparseDispatcher
        """
        self._gates = gates
        self._num_experts = num_experts

        where = tf.to_int32(tf.where(tf.transpose(gates) > 0))
        self._expert_index, self._batch_index = tf.unstack(
            where, num=2, axis=1)
        self._part_sizes_tensor = tf.reduce_sum(tf.to_int32(gates > 0), [0])
        self._nonzero_gates = tf.gather(
            tf.reshape(self._gates, [-1]),
            self._batch_index * num_experts + self._expert_index)

    @add_name_scope()
    def dispatch(self, inp):
        """Create one input Tensor for each expert.

        The 'Tensor' for a expert 'i' contains the slices of 'inp' corresponding
        to the batch elements 'b' where 'gates[b, i] > 0'.

        Args:
          inp: a 'Tensor' of shape "[batch_size, <extra_input_dims>]'
        Returns:
          a list of 'num_experts' 'Tensor's with shapes
            '[expert_batch_size_i, <extra_input_dims>]'.
        """
        inp = tf.gather(inp, self._batch_index)
        return tf.split(inp, self._part_sizes_tensor, 0, num=self._num_experts)

    @add_name_scope()
    def combine(self, expert_out, multiply_by_gates=True):
        """Sum together the expert output, weightd by the gates.

        The slice corresponding to a particular batch element 'b' is computed
        as the sum over all experts 'i' of the expert ouptut, weightd by the
        corresponding gate values. If 'multiply_by_gates' is set to False, the
        gate values are ignored.

        Args:
          expert_out: a list of 'num_experts' 'Tensor's each with shape
            '[expert_batch_size_i, <extra_output_dims>]'.
          multiply_by_gates: a boolean

        Returns:
          a 'Tensor' with shape '[batch_size, <extra_output_dims>]'.
        """
        # see comments on convert_gradient_to_tensor
        stitched = convert_gradient_to_tensor(
            tf.concat(expert_out, 0))
        if multiply_by_gates:
            stitched *= tf.expand_dims(self._nonzero_gates, 1)
        combined = tf.unsorted_segment_sum(stitched, self._batch_index,
                                           tf.shape(self._gates)[0])
        return combined

    def expert_to_gates(self):
        """Gate values corresponding to the examples in the per-expert 'Tensor's.

        Returns:
          a list of 'num_experts' one-dimensional 'Tensor's with type 'tf.float32'
            and shapes '[expert_batch_size_i]'
        """
        return tf.split(
            self._nonzero_gates, self._part_sizes_tensor, 0, num=self._num_experts)

    def expert_to_batch_indices(self):
        """Batch indices corresponding to the examples in the per-expert 'Tensor's.

        Returns:
          a list of 'num_experts' one-dimensional 'Tensor's with type 'tf.int64'
            and shapes '[expert_batch_size_i]'
        """
        return tf.split(
            self._batch_index, self._part_sizes_tensor, 0, num=self._num_experts)

    @property
    def part_sizes(self):
        return self._part_sizes_tensor


class LshGating(object):
    """Class to split key/queries into separate buckets."""

    def __init__(self, depth, nb_hyperplanes, nb_replicat=1, trainable=False):
        """Construct the gating function parameters.

        Compute the gates for a single head.

        Args:
          depth (int): Dimension of the key/queries to dispatch
          nb_hyperplanes (int): Nb of vectors use to split the space. Will determine
            the number of buckets (2^nb_hyperplanes - 1).
          nb_replicat (int): Redundancy to avoid the edge cases (to be in one bucket
            the input should be in a majority)
          trainable (bool): If True, a balance loss is added to force the hyperplane
            to divide the key/query space evenly
        """
        self.depth = depth
        self.nb_hyperplanes = nb_hyperplanes
        self.nb_buckets = 2**nb_hyperplanes
        self.nb_replicat = nb_replicat  # Unused for now
        self.trainable = trainable  # Unused for now

        self.dispatchers = {}

        assert self.nb_replicat == 1  # For now

        with tf.variable_scope("lsh_gating"):
            # Vectors defining the hyperplanes
            self.t_vectors = tf.get_variable(
                "vector",
                shape=(self.depth, self.nb_hyperplanes * self.nb_replicat),
                dtype=tf.float32,
                trainable=self.trainable,
            )
            # Projection vector from the bit space to similarity score space
            self.t_group = tf.constant(
                [self._idx_to_bits(i) for i in range(self.nb_buckets)],
                dtype=tf.float32,
                name="group")

    def _idx_to_bits(self, i):
        """Convert an group index to its bit representation."""
        bits = bin(i)[2:].zfill(self.nb_hyperplanes)  # Pad the bits str with 0
        return [-1.0 if b == "0" else 1.0 for b in bits]

    @add_name_scope("lsh_gating")
    def get_gates(self, x):
        """Return the bucket id of the given tensor.

        Args:
          x (tf.Tensor): float32 of shape [length, depth]

        Returns:
          tf.Tensor: One-hot vector int64 of shape [heads, length, nb_buckets]
            containing the id of the bucket
        """

        # The balance loss don't propagate to the rest of the network
        x = tf.stop_gradient(x)
        # [length, depth] * [depth, nb_vectors * replicat]
        x = tf.matmul(x, self.t_vectors)
        # [length, nb_vector * replicat]
        x = tf.sign(x)  # Get on which side of the hyperplane the keys are.

        # x = tf.reshape(x, [-1, nb_replicat, nb_vector])
        # [length, replicat, nb_vector] * [nb_vector, 2^nb_vector - 1]

        x = tf.matmul(x, self.t_group, transpose_b=True) / self.nb_hyperplanes
        # We get a similarity score for each of the group between [-1, 1]
        # [length, (replicat,) 2^nb_vector - 1]
        # Do an argmax to get the most likely group for each replicat
        x = tf.argmax(x, axis=-1)
        # [length(, replicat)]
        # One-hot for compatibility with the sparse dispatcher
        x = tf.one_hot(x, self.nb_buckets)
        # TODO(epot): Use a loss to force an even distribution
        return x


def should_generate_summaries():
    """Is this an appropiate context to generate summaries.

    Returns:
      a boolean
    """
    if "while/" in tf.contrib.framework.get_name_scope():
        # Summaries don't work well within tf.while_loop()
        return False
    if tf.get_variable_scope().reuse:
        # Avoid generating separate summaries for different data shards
        return False
    return True

# experts


def _my_top_k(x, k):
    """GPU-compatible version of top-k that works for very small constant k.

    Calls argmax repeatedly.

    tf.nn.top_k is implemented for GPU, but the gradient, sparse_to_dense,
    seems not to be, so if we use tf.nn.top_k, then both the top_k and its
    gradient go on cpu. Once this is not an issue, this function becomes
    obsolete and should be replaced by tf.nn.top_k.

    Args:
      x: a 2d Tensor.
      k: a small integer.

    Returns:
      values: a Tensor of shape [batch_size, k]
      indices: a int32 Tensor of shape [batch_size, k]
    """
    if k > 10:
        return tf.nn.top_k(x, k)
    values = []
    indices = []
    depth = tf.shape(x)[1]
    for i in range(k):
        values.append(tf.reduce_max(x, 1))
        argmax = tf.argmax(x, 1)
        indices.append(argmax)
        if i + 1 < k:
            x += tf.one_hot(argmax, depth, -1e9)
    return tf.stack(values, axis=1), tf.to_int32(tf.stack(indices, axis=1))


def _rowwise_unsorted_segment_sum(values, indices, n):
    """UnsortedSegmentSum on each row.

    Args:
      values: a 'Tensor' with shape '[batch_size, k]'
      indices: an integer 'Tensor' with shape '[batch_size, k]'
      n: an integer.
    Returns:
      A 'Tensor' with the same type as 'values' and shape '[batch_size, n]'.
    """
    batch, k = tf.unstack(tf.shape(indices), num=2)
    indices_flat = tf.reshape(
        indices, [-1]) + tf.div(tf.range(batch * k), k) * n
    ret_flat = tf.unsorted_segment_sum(
        tf.reshape(values, [-1]), indices_flat, batch * n)
    return tf.reshape(ret_flat, [batch, n])


def _prob_in_top_k(
        clean_values, noisy_values, noise_stddev, noisy_top_values, k):
    """Helper function to NoisyTopKGating.

    Computes the probabilty that value is in top k, given different random noise.

    This gives us a way of backpropagating from a loss that balances the number
    of times each expert is in the top k experts per example.

    In the case of no noise, pass in None for noise_stddev, and the result will
    not be differentiable.

    Args:
      clean_values: a 'Tensor' of shape [batch, n].
      noisy_values: a 'Tensor' of shape [batch, n]. Equal to clean values plus
        normally distributed noise with standard deviation noise_stddev.
    noise_stddev: a 'Tensor' of shape [batch, n], or None
    noisy_top_values: a 'Tensor' of shape [batch, m].
      "values" Output of tf.top_k(noisy_top_values, m). m >= k+1
    k: an integer.

    Returns:
      a 'Tensor' of shape [batch, n].
    """
    batch = tf.shape(clean_values)[0]
    m = tf.shape(noisy_top_values)[1]
    top_values_flat = tf.reshape(noisy_top_values, [-1])
    # we want to compute the threshold that a particular value would have to
    # exceed in order to make the top k. This computation differs depending
    # on whether the value is already in the top k.
    threshold_positions_if_in = tf.range(batch) * m + k
    threshold_if_in = tf.expand_dims(
        tf.gather(top_values_flat, threshold_positions_if_in), 1)
    is_in = tf.greater(noisy_values, threshold_if_in)
    if noise_stddev is None:
        return tf.to_float(is_in)
    threshold_positions_if_out = threshold_if_in - 1
    threshold_if_out = tf.expand_dims(
        tf.gather(top_values_flat, threshold_positions_if_out), 1)
    # is each value currently in the top k.
    prob_if_in = _normal_distribution_cdf(clean_values - threshold_if_in,
                                          noise_stddev)
    prob_if_out = _normal_distribution_cdf(clean_values - threshold_if_out,
                                           noise_stddev)
    prob = tf.where(is_in, prob_if_in, prob_if_out)
    return prob


def _gates_to_load(gates):
    """Compute the true load per expert, given the gates.

    The load is the number of examples for which the corresponding gate is >0.

    Args:
      gates: a 'Tensor' of shape [batch_size, n]
    Return:
      a float32 'Tensor' of shape [n]
    """
    return tf.reduce_sum(tf.to_float(gates > 0), 0)


def _normal_distribution_cdf(x, stddev):
    """Evaluates the CDF of the normal distribution.

    Normal distribution with mean 0 and standard deviation stddev,
    evaluated at x=x.

    input and output `Tensor`s have matching shapes.

    Args:
      x: a `Tensor`
      stddev: a `Tensor` with the same shape as `x`.

    Returns:
      a `Tensor` with the same shape as `x`.

    """
    return 0.5 * (1.0 + tf.erf(x / (math.sqrt(2) * stddev + 1e-20)))


class Parallelism(object):
    """Helper class for creating sets of parallel function calls.

    The purpose of this class is to replace this code:

        e = []
        f = []
        for i in range(len(devices)):
          with tf.device(devices[i]):
            e_, f_ = func(a[i], b[i], c)
            e.append(e_)
            f.append(f_)

    with this code:

        e, f = expert_utils.Parallelism(devices)(func, a, b, c)
    """

    def __init__(self,
                 device_names_or_functions,
                 reuse=True,
                 caching_devices=None,
                 daisy_chain_variables=False,
                 ps_devices=None):
        """Create a Parallelism.

        Args:
          device_names_or_functions: A list of length n, containing device names
            or device functions (see `tf.device`)
          reuse: True or None.  Whether to reuse variables created in the first
            replica in the subsequent replicas.
          caching_devices: Either `None`, or a list of length n containing device
            names.
          daisy_chain_variables: a boolean - if true, then copies variables in a
            daisy chain between devices.
          ps_devices: list<str>, list of devices for experts.

        Returns:
          a Parallelism.
        """
        assert device_names_or_functions
        self._devices = device_names_or_functions
        self._n = len(device_names_or_functions)
        self._reuse = reuse
        self._caching_devices = self._maybe_repeat(caching_devices)
        self._daisy_chain_variables = daisy_chain_variables
        self._ps_devices = ps_devices or [""]

    def __call__(self, fn, *args, **kwargs):
        """A parallel set of function calls (using the specified devices).

        Args:
          fn: a function or a list of n functions.
          *args: additional args.  Each arg should either be not a list, or a list
             of length n.
          **kwargs: additional keyword args.  Each arg should either be not a
             list, or a list of length n.

        Returns:
          either a single list of length n (if fn does not return a tuple), or a
          tuple of lists of length n (if fn returns a tuple).
        """
        # Construct lists or args and kwargs for each function.
        if args:
            my_args = transpose_list_of_lists(
                [self._maybe_repeat(arg) for arg in args])
        else:
            my_args = [[] for _ in range(self.n)]
        my_kwargs = [{} for _ in range(self.n)]
        for k, v in six.iteritems(kwargs):
            vals = self._maybe_repeat(v)
            for i in range(self.n):
                my_kwargs[i][k] = vals[i]

        # Construct lists of functions.
        fns = self._maybe_repeat(fn)

        # Now make the parallel call.
        outputs = []
        cache = {}
        tensor_to_var = {}
        for i in range(self.n):

            def daisy_chain_getter(getter, name, *args, **kwargs):
                """Get a variable and cache in a daisy chain."""
                device_var_key = (self._devices[i], name)
                if device_var_key in cache:
                    # if we have the variable on the correct device, return it.
                    return cache[device_var_key]
                if name in cache:
                    # if we have it on a different device, copy it from the last device
                    last_device_v = cache[name]
                    var = tensor_to_var[last_device_v]
                    v = tf.identity(last_device_v)
                else:
                    var = getter(name, *args, **kwargs)
                    # v = tf.identity(var._ref())  # pylint: disable=protected-access
                    v = var.read_value()

                # keep track of the original variable
                tensor_to_var[v] = var
                _add_variable_proxy_methods(tensor_to_var[v], v)
                # update the cache
                cache[name] = v
                cache[device_var_key] = v
                return v

            # Variable scope will not reset caching_device on reused variables,
            # so we make a custom getter that uses identity to cache the variable.
            # pylint: disable=cell-var-from-loop
            def caching_getter(getter, name, *args, **kwargs):
                """Cache variables on device."""
                key = (self._caching_devices[i], name)
                if key in cache:
                    return cache[key]

                v = getter(name, *args, **kwargs)
                with tf.device(self._caching_devices[i]):
                    # ret = tf.identity(v._ref())  # pylint: disable=protected-access
                    ret = v.read_value()
                _add_variable_proxy_methods(v, ret)
                cache[key] = ret
                return ret

            if self._daisy_chain_variables:
                custom_getter = daisy_chain_getter
            elif self._caching_devices[i]:
                custom_getter = caching_getter
            else:
                custom_getter = None
            # pylint: enable=cell-var-from-loop
            with tf.name_scope("parallel_%d" % i):
                with tf.variable_scope(
                        tf.get_variable_scope() if self._reuse else "parallel_%d" % i,
                        reuse=True if i > 0 and self._reuse else None,
                        caching_device=self._caching_devices[i],
                        custom_getter=custom_getter):
                    # TODO(noam, epot, avaswani)
                    # Allows for passing no device in case you want to default to the
                    # existing device. This is needed when we put all experts on a single
                    # device, for example in local_moe.
                    if self._devices[i] != DEFAULT_DEV_STRING:
                        with tf.device(self._devices[i]):
                            outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
                    else:
                        outputs.append(fns[i](*my_args[i], **my_kwargs[i]))
        if isinstance(outputs[0], tuple):
            outputs = list(zip(*outputs))
            outputs = tuple([list(o) for o in outputs])
        return outputs

    @property
    def n(self):
        return self._n

    @property
    def devices(self):
        return self._devices

    @property
    def ps_devices(self):
        return self._ps_devices

    def _maybe_repeat(self, x):
        """Utility function for processing arguments that are singletons or lists.

        Args:
          x: either a list of self.n elements, or not a list.

        Returns:
          a list of self.n elements.
        """
        if isinstance(x, list):
            assert len(x) == self.n
            return x
        else:
            return [x] * self.n


def transpose_list_of_lists(lol):
    """Transpose a list of equally-sized python lists.

    Args:
      lol: a list of lists
    Returns:
      a list of lists
    """
    assert lol, "cannot pass the empty list"
    return [list(x) for x in zip(*lol)]


def _add_variable_proxy_methods(var, proxy_tensor):
    """Proxy methods of underlying variable.

    This enables our custom getters to still work with, e.g., batch norm.

    Args:
      var: Variable to proxy
      proxy_tensor: Tensor that is identity of var
    """
    proxy_tensor.read_value = lambda: tf.identity(proxy_tensor)
    proxy_tensor.assign_sub = var.assign_sub
    proxy_tensor.assign = var.assign
    proxy_tensor.initialized_value = var.initialized_value


def cast_like(x, y):
    """Cast x to y's dtype, if necessary."""
    x = tf.convert_to_tensor(x)
    y = tf.convert_to_tensor(y)

    if x.dtype.base_dtype == y.dtype.base_dtype:
        return x

    cast_x = tf.cast(x, y.dtype)
    if cast_x.device != x.device:
        tf.logging.warning("cast for %s may induce copy from '%s' to '%s'",
                           x.name, x.device, cast_x.device)
    return cast_x


def dropout_with_broadcast_dims(x, keep_prob, broadcast_dims=None, **kwargs):
    """Like tf.nn.dropout but takes broadcast_dims instead of noise_shape.

    Instead of specifying noise_shape, this function takes broadcast_dims -
    a lost of dimension numbers in which noise_shape should be 1. The random
    keep/drop tensor has dimensionality 1 along these dimensions.

    Args:
      x: a floating point tensor.
      keep_prob: A scalar Tensor with the same type as x.
        The probabilty that each element is kept.
      broadcast_dims: an optional list of integers
        the dimensions along which to boradcast the keep/drop flasg.
      **kwargs: keyword arguments to tf.nn.dropout other than "noise_shape".
    Returns:
      A Tensor with the same size and shape as .
    """
    assert "noise_shape" not in kwargs
    if broadcast_dims:
        shape = tf.shape(x)
        ndims = len(x.get_shape())
        # Allow dimensions like "-1" as well.
        broadcast_dims = [dim + ndims if dim <
                          0 else dim for dim in broadcast_dims]
        kwargs["noise_shape"] = [
            1 if i in broadcast_dims else shape[i] for i in range(ndims)]
        return tf.nn.dropout(x, keep_prob, **kwargs)

#################         ATENTION UTILS           ################


def reshape_by_blocks(x, x_shape, memory_block_size):
    x = tf.reshape(x, [
        x_shape[0], x_shape[1], x_shape[2] // memory_block_size,
        memory_block_size, x_shape[3]
    ])
    return x


@add_name_scope()
def embedding_to_padding(emb):
    """Calculates the padding mask based on which embeddings are all zero.

    We have hacked symbol_modality to return all-zero embeddings for padding.

    Args:
      emb: a Tensor with shape [..., depth].
    Returns:
      a float Tensor with shape [...].
    """
    emb_sum = tf.reduce_sum(tf.abs(emb), axis=-1)
    return tf.to_float(tf.equal(emb_sum, 0.0))


@add_name_scope()
def split_last_dimension(x, n):
    """Reshape x so that the last dimension becomes two dimensions.

    The first of these two dimensions is n.

    Args:
      x: a Tensor with shape [..., m]
      n: an integer.

    Returns:
      a Tensor with shape [..., n, m/n]
    """
    x_shape = shape_list(x)
    m = x_shape[-1]
    if isinstance(m, int) and isinstance(n, int):
        assert m % n == 0
    return tf.reshape(x, x_shape[:-1] + [n, m // n])


@add_name_scope()
def split_heads(x, num_heads):
    """Split channels (dimension 2) into multiple heads (becomes dimension 1).

    Args:
      x: a Tensor with shape [batch, length, channels]
      num_heads: an integer

    Returns:
      a Tensor with shape [batch, num_heads, length, channels / num_heads]
    """
    return tf.transpose(split_last_dimension(x, num_heads), [0, 2, 1, 3])


def compute_attention_component(antecedent,
                                total_depth,
                                filter_width=1,
                                padding="VALID",
                                name="c"):
    """Computes attention compoenent (query, key or value).

    Args:
      antecedent: a Tensor with shape [batch, length, channels]
      total_depth: an integer
      filter_width: An integer specifying how wide you want the attention
        component to be.
      padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
      name: a string specifying scope name.

    Returns:
      c : [batch, length, depth] tensor
    """
    if filter_width == 1:
        return tf.layers.dense(
            antecedent, total_depth, use_bias=False, name=name)
    else:
        return conv1d(
            antecedent, total_depth, filter_width, padding, name=name)


def _generate_relative_positions_matrix(length, max_relative_position):
    """Generates matrix of relative positions between inputs."""
    range_vec = tf.range(length)
    range_mat = tf.reshape(tf.tile(range_vec, [length])), [length, length]
    distance_mat = range_mat - tf.transpose(range_mat)
    distance_mat_clipped = tf.clip_by_value(
        distance_mat, -max_relative_position, max_relative_position)
    # Shift values to be >= 0. Each integer still uniquely identifies a relative position difference.
    final_mat = distance_mat_clipped + max_relative_position
    return final_mat


def _generate_relative_positions_embeddings(length, depth, max_relative_position, name):
    """Generates tensor of size [length, length, depth]."""
    with tf.variable_scope(name):
        relative_positions_matrix = _generate_relative_positions_matrix(
            length, max_relative_position)
        vocab_size = max_relative_position * 2 + 1
        embeddings_table = tf.get_variable("embeddings", [vocab_size, depth])
        embeddings = tf.gather(embeddings_table, relative_positions_matrix)
        return embeddings


def _relative_attention_inner(x, y, z, transpose):
    """Relative position-aware dot-product attention inner calculation.

    This batches matrix multiply calculations to avoid unnecessary broadcasting.

    Args:
      x: Tensor with shape [batch_size, heads, length, length or depth].
      y: Tensor with shape [batch_size, heads, length, depth].
      z: Tensor with shape [length, length, depth].
      transpose: Whether to transpose inner matrices of y and z. Should be true if
          last dimension of x is depth, not length.

    Returns:
      A Tensor with shape [batch_size, heads, length, length or depth].
    """
    batch_size = tf.shape(x)[0]
    heads = x.get_shape().as_list()[1]
    length = tf.shape(x)[2]

    # xy_matmul is [batch_size, heads, length, length or depth]
    xy_matmul = tf.matmul(x, y, transpose_b=transpose)
    # x_t is [length, batch_size, heads, length or depth]
    x_t = tf.transpose(x, [2, 0, 1, 3])
    # x_t_r is [length, batch_size * heads, length or depth]
    x_t_r = tf.reshape(x_t, [length, heads * batch_size, -1])
    # x_tz_matmul is [length, batch_size * heads, length or depth]
    x_tz_matmul = tf.matmul(x_t_r, z, transpose_b=transpose)
    # x_tz_matmul_r is [length, batch_size, heads, length or depth]
    x_tz_matmul_r = tf.reshape(x_tz_matmul, [length, batch_size, heads, -1])
    # x_tz_matmul_r_t is [batch_size, heads, length, length or depth]
    x_tz_matmul_r_t = tf.transpose(x_tz_matmul_r, [1, 2, 0, 3])
    return xy_matmul + x_tz_matmul_r_t


def gather_dilated_memory_blocks(x,
                                 num_memory_blocks,
                                 gap_size,
                                 query_block_size,
                                 memory_block_size,
                                 gather_indices,
                                 direction="left"):
    """Gathers blocks with gaps in between.

    Args:
      x: A tensor of shape [length, batch, heads, depth]
      num_memory_blocks:     num_memory_blocks: how many memory blocks to look
        in "direction". Each will be separated by gap_size.
      gap_size: an integer indicating the gap size
      query_block_size: an integer indicating size of query block
      memory_block_size: an integer indicating the size of a memory block.
      gather_indices: The indices to gather from.
      direction: left or right
    Returns:
      a tensor of shape [batch, heads, blocks, block_length, depth]
    """

    gathered_blocks = []
    # gathering memory blocks
    for block_id in range(num_memory_blocks):
        block_end_index = -(query_block_size + gap_size *
                            (block_id + 1) + memory_block_size * block_id) - 1
        block_start_index = (
            (memory_block_size + gap_size) * (num_memory_blocks - (block_id + 1)))
        if direction != "left":
            [block_end_index,
             block_start_index] = [-block_start_index - 1, -block_end_index + 1]

        def gather_dilated_1d_blocks(x, gather_indices):
            x_new = tf.gather(x, gather_indices)
            # [batch, heads, blocks, block_length, dim]
            return tf.transpose(x_new, [2, 3, 0, 1, 4])

        gathered_blocks.append(
            gather_dilated_1d_blocks(x[block_start_index:block_end_index],
                                     gather_indices))
    return tf.concat(gathered_blocks, 3)


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None):
    """Matrix band part of ones."""
    if all([isinstance(el, int) for el in [rows, cols, num_lower, num_upper]]):
        # Needed info is constant, so we construct in numpy
        if num_lower < 0:
            num_lower = rows - 1
        if num_upper < 0:
            num_upper = cols - 1
        lower_mask = np.tri(cols, rows, num_lower).T
        upper_mask = np.tri(rows, cols, num_upper)
        band = np.ones((rows, cols)) * lower_mask * upper_mask
        if out_shape:
            band = band.reshape(out_shape)
        band = tf.constant(band, tf.float32)
    else:
        band = tf.matrix_band_part(tf.ones([rows, cols]),
                                   tf.cast(num_lower, tf.int64),
                                   tf.cast(num_upper, tf.int64))
        if out_shape:
            band = tf.reshape(band, out_shape)

    return band


@add_name_scope()
def attention_bias_local(length, max_backward, max_forward):
    """Create an bias tensor to be added to attention logits.

    A position may attend to positions at most max_distance from it,
    forward and backwards.

    This does not actually save any computation.

    Args:
        length: int
        max_backward: int, maximum distance backward to attend. Negative values
        indicate unlimited.
        max_forward: int, maximum distance forward to attend. Negative values
        indicate unlimited.

    Returns:
        a `Tensor` with shape [1, 1, length, length].
    """
    band = ones_matrix_band_part(
        length,
        length,
        max_backward,
        max_forward,
        out_shape=[1, 1, length, length])
    return -1e9 * (1.0 - band)


@add_name_scope()
def attention_bias_lower_triangle(length):
    """Create an bias tensor to be added to attention logits.

    Allows a query to attend to all positions up to and including its own.

    Args:
    length: a Scalar.

    Returns:
        a `Tensor` with shape [1, 1, length, length].
    """
    return attention_bias_local(length, -1, 0)


def _absolute_position_to_relative_position_masked(x):
    """Helper to dot_product_self_attention_relative_v2.

    Rearrange an attention logits or weights Tensor.

    The dimensions of the input represent:
    [batch, heads, query_position, memory_position]

    The dimensions of the output represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]

    Only works with masked_attention.  Undefined behavior for regions of the
    input where memory_position > query_position.

    Args:
      x: a Tensor with shape [batch, heads, length, length]

    Returns:
      a Tensor with shape [batch, heads, length, length]
    """
    batch, heads, length, _ = shape_list(x)
    x = tf.pad(x, [[0, 0], [0, 0], [1, 0], [0, 0]])
    x = tf.reshape(x, [batch, heads, length, length + 1])
    x = tf.slice(x, [0, 0, 0, 1], [batch, heads, length, length])
    return x


def _relative_position_to_absolute_position_masked(x):
    """Helper to dot_product_self_attention_relative_v2.

    Rearrange an attention logits or weights Tensor.

    The dimensions of the input represent:
    [batch, heads, query_position, memory_position - query_position + length - 1]

    The dimensions of the output represent:
    [batch, heads, query_position, memory_position]

    Only works with masked_attention.  Undefined behavior for regions of the
    input where memory_position > query_position.

    Args:
      x: a Tensor with shape [batch, heads, length, length]

    Returns:
      a Tensor with shape [batch, heads, length, length]
    """
    batch, heads, length, _ = shape_list(x)
    x = tf.pad(x, [[0, 0], [0, 0], [0, 0], [1, 0]])
    x = tf.reshape(x, [batch, heads, 1 + length, length])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    return x


def compute_qkv(query_antecedent,
                memory_antecedent,
                total_key_depth,
                total_vale_depth,
                q_filter_width=1,
                kv_filter_width=1,
                q_padding="VALID",
                kv_padding="VALID"):
    """Computes query, key and value.

    Args:
      query_antecedent: a Tensor with shape [batch, length_q, channels]
      memory_antecedent: a Tensor with shape [batch, length_m, channels]
      total_key_depth: an integer
      total_value_depth: an integer
      q_filter_width: an integer specifying how wide you want the query to be.
      kv_filter_width: an integer specifying how wide you want the keys and values
                      to be.
      q_padding: one of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
      kv_padding: one of "VALID", "SAME" or "LEFT". Default is VALID: No padding.

    Returns:
      q, k, v: [batch, length, dept] tensors
    """
    if memory_antecedent is None:
        memory_antecedent = query_antecedent
    q = compute_attention_component(
        query_antecedent, total_key_depth, q_filter_width, q_padding, "q")
    k = compute_attention_component(
        memory_antecedent, total_key_depth, kv_filter_width, kv_padding, "k")
    v = compute_attention_component(
        memory_antecedent, total_vale_depth, kv_filter_width, kv_padding, "v")
    return q, k, v
