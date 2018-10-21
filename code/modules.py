import tensorflow as tf
import numpy as np

import functools
import six

from dnc.util import add_name_scope
from dnc.util import flatten_all_but_last, reshape_like, cv_squared, coordinate_tensor, shape_list
from dnc.util import SparseDispatcher, Parallelism, should_generate_summaries
from dnc.util import _my_top_k, _rowwise_unsorted_segment_sum, _prob_in_top_k, _gates_to_load
from dnc.util import compute_qkv, compute_attention_component, split_heads, combine_heads, map_ids
from dnc.attentions import dot_product_attention, dot_product_attention_relative, dot_product_self_attention_relative_v2
from dnc.attentions import masked_within_block_local_attention_1d, masked_local_attention_1d, local_attention_1d
from dnc.attentions import masked_dilated_self_attention_1d, dilated_self_attention_1d, attention_bias_lower_triangle

# REMOVE PAD AND RESTORE PAD ATTENTION LM MOE CHECK
DEFAULT_DEV_STRING = "existing_device"


def normalize(inputs, epsilon=1e-8, scope="ln", reuse=None):
    """Applies layer normalization.

      Args:
        inputs: A tensor with 2 or more dimensions, where the first dimension has
          'batch_size'.
        epsilon: A floating number. A very small number for preventing ZeroDivision Error.
        scope: Optional scope for 'variable_scope'.
        reuse: Boolean, whether to reuse the weights of the previous layer
          by the same name.
      Returns:
        A tensor with the same shape and data dtype as 'inputs'.
    """
    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def positional_encoding(inputs,
                        num_units,
                        zero_pad=True,
                        scale=True,
                        scope="positional_encoding",
                        reuse=None):
    """Sinusoidal positional encoding.

    Args:
      inputs: A 2d Tensor with shape of (N, T).
      num_units: Output dimensionality
      zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
      scale: Boolean. If True, the output will be multiplied by sqrt num_units(check details from paper)
      scope: Optional scope for 'variable_scope'.
      reuse: Boolean, whether to reuse the weigths of a previous layer
        by the same name

      Returns:
        A 'Tensor' with one more rank thank inputs, with the dimensionality should be 'num_units'
    """

    N, T = inputs.get_shape().as_list()
    with tf.variable_scope(scope, reuse=reuse):
        positional_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])

        # First part of the PE function: sin and cos argument
        positional_enc = np.array([
            [pos / np.power(10000, 2.*i/num_units) for i in range(num_units)]
            for pos in range(T)])

        # Second part, apply the cosine to even columns and sin to odds.
        positional_enc[:, 0::2] = np.sin(positional_enc[:, 0::2])  # dim 2i
        positional_enc[:, 1::2] = np.cos(positional_enc[:, 1::2])  # dim 2i+1

        # Convert to a tensor
        lookup_table = tf.convert_to_tensor(positional_enc)

        if zero_pad:
            lookup_table = tf.concat((tf.zeros(shape=[1, num_units]),
                                      lookup_table[1:, :]), 0)
        outputs = tf.nn.embedding_lookup(lookup_table, positional_ind)

        if scale:
            outputs = outputs * num_units**0.5
        return outputs


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        bias,
                        total_key_depth,
                        total_value_depth,
                        output_depth,
                        num_heads,
                        dropout_rate,
                        max_relative_position=None,
                        image_shapes=None,
                        attention_type="dot_product",
                        block_length=128,
                        block_width=128,
                        q_filter_width=1,
                        kv_filter_width=1,
                        q_padding="VALID",
                        kv_padding="VALID",
                        cache=None,
                        gap_size=0,
                        num_memory_blocks=2,
                        name="multihead_attention",
                        save_weights_to=None,
                        make_image_summary=True,
                        dropout_broadcast_dims=None,
                        max_length=None,
                        **kwargs):
    """Multihead scaled-dot-product attention with input/output transformations.

    Args:
        query_antecedent: a Tensor with shape [batch, length_q, channels]
        memory_antecedent: a Tensor with shape [batch, length_m, channels] or None
        bias: bias Tensor (see attention_bias())
        total_key_depth: an integer
        total_value_depth: an integer
        output_depth: an integer
        num_heads: an integer dividing total_key_depth and total_value_depth
        dropout_rate: a floating point number
        max_relative_position: Maximum distance between inputs to generate
                            unique relation embeddings for. Only relevant
                            when using "dot_product_relative" attention.
        image_shapes: optional tuple of integer scalars.
                    see comments for attention_image_summary()
        attention_type: a string, either "dot_product", "dot_product_relative",
                        "local_mask_right", "local_unmasked", "masked_dilated_1d",
                        "unmasked_dilated_1d" or any attention function with the
                        signature (query, key, value, **kwargs)
        block_length: an integer - relevant for "local_mask_right"
        block_width: an integer - relevant for "local_unmasked"
        q_filter_width: An integer specifying how wide you want the query to be.
        kv_filter_width: An integer specifying how wide you want the keys and values
                        to be.
        q_padding: One of "VALID", "SAME" or "LEFT". Default is VALID: No padding.
                kv_padding: One of "VALID", "SAME" or "LEFT". Default is "VALID":
                no padding.
        cache: dict containing Tensors which are the results of previous
            attentions, used for fast decoding. Expects the dict to contrain two
            keys ('k' and 'v'), for the initial call the values for these keys
            should be empty Tensors of the appropriate shape.
                'k' [batch_size, 0, key_channels]
                'v' [batch_size, 0, value_channels]
        gap_size: Integer option for dilated attention to indicate spacing between
                memory blocks.
        num_memory_blocks: Integer option to indicate how many memory blocks to look
                        at.
        name: an optional string.
        save_weights_to: an optional dictionary to capture attention weights
        for vizualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
        make_image_summary: Whether to make an attention image summary.
        dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.
        max_length: an integer - needed by relative attention
        **kwargs (dict): Parameters for the attention function

    Caching:
        WARNING: For decoder self-attention, i.e. when memory_antecedent == None,
        the caching assumes that the bias contains future masking.

        The caching works by saving all the previous key and value values so that
        you are able to send just the last query location to this attention
        function. I.e. if the cache dict is provided it assumes the query is of the
        shape [batch_size, 1, hiddem_dim] rather than the full memory.

    Returns:
        The result of the attention transformation. The output shape is
            [batch_size, length_q, hidden_dim]
        unless the cache dict is provided in which case only the last memory
        position is calculated and the output shape is [batch_size, 1, hidden_dim]
        Optionally returns an additional loss parameters (ex: load balance loss for
        the experts) returned by the attention_type function.

    Raises:
        ValueError: if the key depth or value depth are not divisible by the
        number of attention heads.
    """
    if total_key_depth % num_heads != 0:
        raise ValueError("Key depth (%d) must be divisible by the number "
                         "attention heads (%d)." % (total_key_depth, num_heads))
    if total_value_depth % num_heads != 0:
        raise ValueError("Value depth (%d) must be divisible by the number of "
                         "attention heads (%d)." % (total_value_depth, num_heads))
    with tf.variable_scope(name, default_name="multihead_attention",
                           values=[query_antecedent, memory_antecedent]):

        if cache is None or memory_antecedent is None:
            q, k, v = compute_qkv(query_antecedent, memory_antecedent,
                                  total_key_depth, total_value_depth, q_filter_width,
                                  kv_filter_width, q_padding, kv_padding)
        if cache is not None:
            if attention_type != "dot_product":
                # TODO: support caching when using relative position
                # representations, i.e. "dot_product_relative" attention.
                raise NotImplementedError(
                    "Caching is not guaranteed to work with attention types other than"
                    " dot_product.")
                if bias is None:
                    raise ValueError("Bias required for caching. See function docstring "
                                     "for details.")

                if memory_antecedent is not None:
                    # Encoder-Decoder Attention Cache
                    q = compute_attention_component(
                        query_antecedent, total_key_depth, q_filter_width, q_padding, "q")
                    k = cache["k_encdec"]
                    v = cache["v_encdec"]
                else:
                    k = split_heads(k, num_heads)
                    v = split_heads(v, num_heads)
                    k = cache["k"] = tf.concat([cache["k"], k], axis=2)
                    v = cache["v"] = tf.concat([cache["v"], v], axis=2)

                q = split_heads(q, num_heads)
                if cache is None:
                    k = split_heads(k, num_heads)
                    v = split_heads(v, num_heads)

                key_depth_per_head = total_key_depth // num_heads
                q *= key_depth_per_head**-0.5

                additional_returned_value = None
                if callable(attention_type):    # Generic way to extend multihead_attention
                    x = attention_type(q, k, v, **kwargs)
                    if isinstance(x, tuple):
                        x, additional_returned_value = x  # Unpack
                elif attention_type == "dot_product":
                    x = dot_product_attention(
                        q,
                        k,
                        v,
                        bias,
                        dropout_rate,
                        image_shapes,
                        save_weigths_to=save_weights_to,
                        dropout_broadcast_dims=dropout_broadcast_dims)
                elif attention_type == "dot_product_relative":
                    x = dot_product_attention_relative(
                        q,
                        k,
                        v,
                        bias,
                        max_relative_position,
                        dropout_rate,
                        image_shapes)
                elif attention_type == "dot_product_relative_v2":
                    x = dot_product_self_attention_relative_v2(
                        q,
                        k,
                        v,
                        bias,
                        max_length,
                        dropout_rate,
                        image_shapes,
                        dropout_broadcast_dims=dropout_broadcast_dims)
                elif attention_type == "local_within_block_mask_right":
                    x = masked_within_block_local_attention_1d(
                        q, k, v, block_length=block_length)
                elif attention_type == "local_mask_right":
                    x = masked_local_attention_1d(
                        q,
                        k,
                        v,
                        block_length=block_length)
                elif attention_type == "local_unmasked":
                    x = local_attention_1d(
                        q, k, v, block_length=block_length, filter_width=block_width)
                elif attention_type == "masked_dilated_1d":
                    x = masked_dilated_self_attention_1d(q, k, v, block_length, block_width,
                                                         gap_size, num_memory_blocks)
                else:
                    assert attention_type == "unmasked_dilated_1d"
                    x = dilated_self_attention_1d(
                        q, k, v, block_length, block_width, gap_size, num_memory_blocks)
                x = combine_heads(x)

                # Set last dim specifically.
                x.set_shape(x.shape.as_list()[:-1] + [total_value_depth])

                x = tf.layers.dense(
                    x, output_depth, use_bias=False, name="output_transform")
                if additional_returned_value is not None:
                    return x, additional_returned_value
                return x


def feedforward(inputs,
                num_units=[2048, 512],
                scope="multihead_attention",
                reuse=None):
    """Point-wise feed forward net.

    Args:
      inputs: a 3d tensor with shape of[N, T, C]
      num_units: A list of two intergers.
      scope: Optional scope for 'variable_scope'.
      reuse: Boolean, whether to reuse the weigths of a previous layer
        by the same name.

      Returns:
        A 3d tensor with the same shape and dtype as inputs
    """
    with tf.variable_scope(scope, reuse=reuse):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = normalize(outputs)


def label_smoothing(inputs, epsilon=0.1):
    """Applies label smoothing. See https: // arxiv.org/abs/1512.00567.

    Args:
      inputs: A 3d tensor with shape of[N, T, V], where V is the number of vocabulary.
      epsilon: Smoothing rate.

   For example,

    ```
    import tensorflow as tf
    inputs = tf.convert_to_tensor([[[0, 0, 1], 
       [0, 1, 0],
       [1, 0, 0]],

      [[1, 0, 0],
       [1, 0, 0],
       [0, 1, 0]]], tf.float32)

    outputs = label_smoothing(inputs)

    with tf.Session() as sess:
        print(sess.run([outputs]))

    >>
    [array([[[ 0.03333334,  0.03333334,  0.93333334],
        [ 0.03333334,  0.93333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334]],

       [[ 0.93333334,  0.03333334,  0.03333334],
        [ 0.93333334,  0.03333334,  0.03333334],
        [ 0.03333334,  0.93333334,  0.03333334]]], dtype=float32)]   
    ```    
    """
    K = inputs.get_shape().as_list()[-1]  # number of channels
    return ((1-epsilon) * inputs) + (epsilon / K)


def noisy_top_k_gating(x,
                       num_experts,
                       train,
                       k=2,
                       initializer=tf.zeros_initializer(),
                       noisy_gating=True,
                       noise_epsilon=1e-2,
                       name=None):
    """Noisy top-k gating.

    See paper : https://arxiv.org/abs/1701.06538.

    Args:
      x: input Tensor with shape [batch_size, input_size]
      num_experts: an integer
      train: a boolean - we only add noise at training time.
      k: an integer - number of experts per example
      initializer: an initializer
      noisy_gating: a boolean
      noisy_epsilon: a float
      name: an optional string

    Returns:
      gates: a Tensor with shape [batch_size, num_experts]
      load: a Tensor with shape [num_experts]
    """
    with tf.variable_scope(name, default_name="noisy_top_k_gating"):
        input_size = x.get_shape().as_list()[-1]
        w_gate = tf.get_variable(
            "w_gate", [input_size, num_experts], tf.float32, initializer)
        if noisy_gating:
            w_noise = tf.get_variable("w_noise",
                                      [input_size, num_experts], tf.float32,
                                      initializer)
        clean_logits = tf.matmul(x, w_gate)
        if noisy_gating:
            raw_noise_stddev = tf.matmul(x, w_noise)
            noise_stddev = ((tf.nn.softplus(raw_noise_stddev) + noise_epsilon) *
                            (tf.to_float(train)))
            noisy_logits = clean_logits + (
                tf.random_normal(tf.shape(clean_logits)) * noise_stddev)
            logits = noisy_logits
            if should_generate_summaries():
                tf.summary.histogram("noisy_logits", noisy_logits)
                tf.summary.histogram("nose_stddev", noise_stddev)
            else:
                logits = clean_logits
            top_logits, top_indices = _my_top_k(
                logits, min(k + 1, num_experts))
            top_k_logits = tf.slice(top_logits, [0, 0], [-1, k])
            top_k_indices = tf.slice(top_indices, [0, 0], [-1, k])
            top_k_gates = tf.nn.softmax(top_k_logits)
            # This will be a 'Tensor' of shape '[batch_size, n]', with zeros in the
            # positions corresponding to all but the top k experts per example.
            gates = _rowwise_unsorted_segment_sum(
                top_k_gates, top_k_indices, num_experts)

            if noisy_gating and k < num_experts:
                load = tf.reduce_sum(
                    _prob_in_top_k(clean_logits, noisy_logits, noise_stddev, top_k_logits, k), 0)
            else:
                load = _gates_to_load(gates)
            if should_generate_summaries():
                tf.summary.histogram("importance", tf.reduce_sum(gates, 0))
                tf.summary.histogram("load", load)
            return gates, load


def local_moe(x,
              train,
              expert_fn,
              num_experts,
              k=2,
              loss_coef=1e-2,
              pass_x=True,
              pass_gates=False,
              additional_dispatch_params=None,
              name=None):
    """Call a local mixture of experts.

    Args:
        x: a tensors with shape [... , input_size]
        train: a boolean scalar.
        expert_fn: a function.
        num_experts: an integer - number of experts
        k: an integer - how many experts to use for each batch element
        loss_coef: a scalar - multiplier on load-balancing losses
        pass_x: a boolean. If true, x will also be dispatched to the experts.
        pass_gates: a boolean. If true, gates will be passed to experts. Might be
        necessary when dealing with sparse encoder-encoder decoder attention
        additional_dispatch_params: The extra tensors that need to be sent to each
        expert. Examples include batch batch coordinates (see
        common_attention.local_expert_attention)
        name: a string

    Returns:
        y: a tensor.  Has the same shape as x, except for the last dimension,
        which is output_size.
        extra_training_loss: a scalar.  This should be added into the overall
        training loss of the model.  The backpropagation of this loss
        encourages all experts to be approximately equally used across a batch.
    """
    with tf.variable_scope(name, default_name="local_moe"):
        x_flat = flatten_all_but_last(x)

        # The gates indicate which batch elemnts go to which tensors.
        # load is a measure of approximately how many examples go to each expert
        gates, load = noisy_top_k_gating(
            x_flat,
            num_experts,
            train,
            k,
            initializer=tf.zeros_initializer(),
            noisy_gating=True,
            noise_epsilon=1e-2)
        # This magic object helps us shuffle data between datashards and experts.
        dispatcher = SparseDispatcher(num_experts, gates)

        # Set up expert_fn arguments
        expert_kwargs = {}
        if pass_x:
            expert_kwargs["x"] = dispatcher.dispatch(x_flat)
        if pass_gates:
            expert_kwargs["gates"] = dispatcher.expert_to_gates()
        for key, val in six.iteritems(additional_dispatch_params or {}):
            val = flatten_all_but_last(val)
            expert_kwargs[key] = dispatcher.dispatch(val)

        ep = Parallelism([DEFAULT_DEV_STRING] * num_experts, reuse=None)
        expert_outputs = ep(expert_fn, **expert_kwargs)

        y_flat = dispatcher.combine(expert_outputs)
        y = reshape_like(y_flat, x)

        importance = tf.reduce_sum(gates, 0)
        loss = loss_coef * (cv_squared(importance) + cv_squared(load))
        return y, loss


def local_expert_attention(x,
                           k,
                           loss_coef,
                           attention_num_experts,
                           train=True,
                           batch_coordinate=None,
                           **kwargs):
    """Attention using a mixture of experts.

    Positions sent to the same expert can attend to each other.
    The mixture of experts is "local" in that it is repplicated on each
    datashard.

    local_moe flatten all batches so to avoid problems with padding(ex: all
    padding going to the same expert, self attention attending to non null
    padding tokens,...), the padding should be removed before.

    Args:
        x: a Tensor with shape [batch, length, depth] or [1, batch*length, depth]
        k: The number of experts to dispatch each example to
        loss_coef: a scalar. A multiplier for the expert loss
        attention_num_experts: The number of experts to use
        train: a boolean for the current mode
        batch_coordinate (tf.Tensor): int32 tensor of shape [1, batch*length, 1]
        containing the batch ids. If None, deduced from first dim of x.
        **kwargs: Arguments to forward to self_attention_expert

    Returns:
        y: a Tensor with shape [batch, length, depth]
        loss: a Scalar
    """
    if batch_coordinate is None:
        batch_coordinate = tf.expand_dims(
            coordinate_tensor(shape_list(x)[:-1], axis=0), axis=-1)
    with tf.variable_scope("local_expert_attention"):
        additional_dispatch_params = {"batch_coordinate": batch_coordinate}
        return local_moe(
            x,
            train,
            functools.partial(self_attention_expert, **kwargs),
            attention_num_experts,
            k=k,
            loss_coef=loss_coef,
            pass_x=True,
            pass_gates=False,
            additional_dispatch_params=additional_dispatch_params)


_expert_count = 0
attention_bias_coordinates = functools.partial(
    attention_bias_batch,
    condition_fn=lambda bias: tf.minimum(1.0, tf.abs(bias)),
)


@add_name_scope()
def attention_bias_batch(
        batch_coordinates_q,
        batch_coordinates_k=None,
        condition_fn=None):
    """Generate a mask to prevent the batch to attend to each others.

    Args:
      batch_coordinates_q (tf.Tensor): int32 of shape [length_q, 1] containing the 
        coordinates of the batches
      batch_coordinates_k (tf.Tensor): int32 of shape [length_k, 1] containing the 
        coordinates of the batches. If None, do self attention (q and k identical)
      condition_fn (fct): A function defining which type of mask build

    Returns:
      tf.Tensor: float32 mask of shape [length_q, length_k] containing either 0 or 
        -infinity (-1e9)
    """
    if batch_coordinates_k is None:
        batch_coordinates_k = batch_coordinates_q

    # Convert to float first because of b/25387198
    def to_float(bc):
        bc = tf.squeeze(bc, 1)
        bc = tf.to_float(bc)
        return bc

    bc_v = tf.expand_dims(to_float(batch_coordinates_q), 1)
    bc_h = tf.expand_dims(to_float(batch_coordinates_k), 0)
    bias_batch = bc_h - bc_v  # Broadcast to create [length_q, length_k] mask.
    # Threshold non zeros to 1.0
    bias_batch = condition_fn(bias_batch)
    bias_batch *= -1e9  # Set non zeros to -infinity
    return bias_batch


def self_attention_expert(x,
                          batch_coordinate,
                          mask_rigth=True,
                          split_batch=False,
                          attention_num_head=1,
                          attention_kq_size=None,
                          attention_v_size=None):
    """Implementing attention that runs inside each expert.

    Args:
      x: A tensor of shape[batch, depth]. Contains representations from different
        positions, which are lexicographically ordered.
      bacth_coordinate: A tensor of shape [batch, 1] containing the batch
        coordinate of each element in x. This is needed to make sure that
        positions from different sequences don't attend to each other.
      mask_right: A bool. If true, we will not attend to positions on the right,
        just as decoder self attention.
      split_batch (bool): If True, each sequence of the batch is processed individually
        on a loop. If False, the sequences are processed all at once and a mask is applied to
        isolate the sequences from each others
      attention_num_head (int): number of attention heads
      attention_kq_size (int): dimension used for the attention key, and query
      attention_v_size (int): dimension used for the attention value

    Returns:
      out: A tensor of shape [batch, depth].
    example use:
    local_moe(
        ...
        expert_fn=functools.partial(self_attention_expert, mask_right=)
    )
    """
    depth = x.get_shape().as_list()[-1]
    length = shape_list(batch_coordinate)[0]

    # Print a warning message if one of the expert isn't used (useful at
    # inference where summaries aren't used and the gating function don't add
    # noise)
    global _expert_count  # Hack to make each expert have a unique id
    _expert_count += 1
    length = tf.cond(
        tf.equal(length, 0),
        lambda: tf.Print(   # pylint: disable=g-long-lambda
            length, [length], "Expert {} empty".format(_expert_count)),
        lambda: length,
    )

    tf.summary.scalar("batch_size", length, family="experts_stats_batch_size")

    attention_kq_size = attention_kq_size or depth
    attention_v_size = attention_v_size or depth

    def length_not_null(x, batch_coordinate):
        """Branch of the graph only evaluated when length isn't null."""

        # Mask between the sequences (not used if map_ids is used)
        bias_batch = attention_bias_coordinates(batch_coordinate)

        def add_or_set_if(prev_bias, new_bias, condition):
            """Add the bias together while considering the None case."""
            if not condition:
                return prev_bias
            if prev_bias is None:
                return new_bias
            return prev_bias + new_bias

        def mask_and_call_attention(x):
            """Function applied once for each sequence of the batch."""

            # Mask to prevent sequences of attending to the future
            length = shape_list(x)[1]  # x has shape [1, length, ...]
            bias_past = tf.reshape(
                attention_bias_lower_triangle(length), [length, length])
            # bias has shape [length, length]

            bias = None
            bias = add_or_set_if(bias, bias_past, mask_rigth)
            bias = add_or_set_if(bias, bias_batch, not split_batch)
            bias = tf.reshape(bias, [1, 1, length, length])

            return multihead_attention(
                x,
                None,
                bias,
                total_key_depth=attention_kq_size,
                total_value_depth=attention_v_size,
                output_depth=depth,
                num_heads=attention_num_head,
                dropout_rate=0.0)

        if split_batch:
            out = map_ids(x, batch_coordinate, mask_and_call_attention)
        else:
            x = tf.reshape(x, [1, length, depth])
            out = mask_and_call_attention(x)
            out = tf.squeeze(out, 0)
        return out

        # If the length is empty, just forward an empty tensor (avoid having to
        # evaluate multihead_attention with tensor having dim equal to zeros)
        out = tf.cond(
            tf.equal(length, 0),
            lambda: tf.zeros(shape=[0, depth],
                             dtype=tf.float32, name="empty_out"),
            lambda: length_not_null(x, batch_coordinate))
        return out
