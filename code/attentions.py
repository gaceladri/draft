from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools
import collections

from code.util import cast_like, dropout_with_broadcast_dims, should_generate_summaries, shape_list
from code.util import _generate_relative_positions_embeddings, _relative_attention_inner, _relative_position_to_absolute_position_masked
from code.util import _absolute_position_to_relative_position_masked, attention_bias_lower_triangle, ones_matrix_band_part
from code.util import gather_dilated_memory_blocks, reshape_by_blocks, embedding_to_padding, flatten_all_but_last
from code.util import add_name_scope, add_var_scope


def dot_product_attention(q,
                          k,
                          v,
                          bias,
                          dropout_rate=0.0,
                          image_shapes=None,
                          name=None,
                          save_weigths_to=None,
                          dropout_broadcast_dims=None):
    """dot-product attention.

    Args:
      q: a Tensor with shape [batch, heads, length_q, depth_k]
      k: a Tensor with shape [batch, heads, length_kv, depth_k]
      v: a Tensor with shape [batch, heads, length_kv, depth_v]
      bias: bias Tensor (see attention_bias())
      dropout_rate: a floating point number
      image_shapes: optional tuple of integer scalars.
        see comments for attention_image_summary()
      name: an optional string
      save_weights_to: an optional dictionary to capture attention weights
        for visualization; the weights tensor will be appended there under
        a string key created from the variable scope (including name).
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.

    Returns:
      A Tensor.
    """
    with tf.variable_scope(
            name, default_name="dot_product_attention", values=[q, k, v]) as scope:
        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)
        if bias is not None:
            bias = cast_like(bias, logits)
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weigths")
        if save_weigths_to is not None:
            save_weigths_to[scope.name] = weights
            save_weigths_to[scope.name + "/logits"] = logits
        # dropping out the attention links for each of the heads
        weights = dropout_with_broadcast_dims(
            weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        return tf.matmul(weights, v)


def dot_product_attention_relative(q,
                                   k,
                                   v,
                                   bias,
                                   max_relative_position,
                                   dropout_rate=0.0,
                                   image_shapes=None,
                                   name=None):
    """Calculate relative position-aware dot-product self-attention.

    The attention calculation is augmented with learned representations from the 
    relative position between each element in q and each element in k and v.

    Args:
      q: a Tensor with shape [batch, heads, length, depth].
      k: a Tensor with shape [batch, heads, length, depth].
      v: a Tensor with shape [batch, heads, length, depth].
      bias: bias Tensor.
      max_relative_position: an integer specifying the maximum distance between
          inputs that unique position embeddings should be learned for.
      dropout_rate: a floating point number.
      image_shapes: optional tuple of integer scalars.
      name: an optional string.

    Returns:
      A Tensor.

    Raises:
      ValueError: if max_relative_position is not > 0.
    """
    if not max_relative_position:
        raise ValueError("Max relative position (%s) should be > 0 when using "
                         "relative self attentnion." % (max_relative_position))
    with tf.variable_scope(
            name, default_name="dot_product_attention_relative", values=[q, k, v]):

        # This calculation only works for self attention.
        # q, k and v must therefore have the same shape.
        q.get_shape().assert_is_compatible_with(k.get_shape())
        q.get_shape().assert_is_compatible_with(v.get_shape())

        # Use separate embeddings suitable for keys and values.
        depth = q.get_shape().as_list()[3]
        lenght = shape_list(q)[2]
        relations_keys = _generate_relative_positions_embeddings(
            lenght, depth, max_relative_position, "relative_positions_keys")
        relations_values = _generate_relative_positions_embeddings(
            lenght, depth, max_relative_position, "relative_positions_vales")

        # Compute self attention considering the relative position embeddings.
        logits = _relative_attention_inner(q, k, relations_keys, True)
        if bias is not None:
            logits += bias
        weigths = tf.nn.softmax(logits, name="attention_weigths")
        weigths = tf.nn.dropout(weigths, 1.0 - dropout_rate)
        return _relative_attention_inner(weigths, v, relations_values, False)


def dot_product_self_attention_relative_v2(q,
                                           k,
                                           v,
                                           bias,
                                           max_length=None,
                                           dropout_rate=0.0,
                                           image_shapes=None,
                                           name=None,
                                           make_image_summary=True,
                                           dropout_broadcast_dims=None):
    """Calculate relative position-aware dot-product self-attention.

    Only works for masked self-attention (no looking forward).
    TODO(noam): extend to unmasked self-attention

    The attention calculation is augmented with learned representations for the
    relative position between each element in q and each element in k and v.

    Args:
      q: a Tensor with shape [batch, heads, length, depth].
      k: a Tensor with shape [batch, heads, length, depth].
      v: a Tensor with shape [batch, heads, length, depth].
      bias: bias Tensor.
      max_length: an integer - changing this invalidates checkpoints
      dropout_rate: a floating point number.
      image_shapes: optional tuple of integer scalars.
      name: an optional string.
      make_image_summary: Whether to make an attention image summary.
      dropout_broadcast_dims:  an optional list of integers less than 4
        specifying in which dimensions to broadcast the dropout decisions.
        saves memory.

    Returns:
      A Tensor.
    """
    with tf.variable_scope(
            name,
            default_name="dot_product_self_attention_relative_v2",
            values=[q, k, v]):

        # This calculation only works for self attention.
        # q, k and v must therefore have the same shape.
        q.get_shape().assert_is_compatible_with(k.get_shape())
        q.get_shape().assert_is_compatible_with(v.get_shape())

        # Use separate embeddings suitable for keys and values.
        length = shape_list(q)[2]
        assert max_length is not None

        # [batch, num_heads, query_length, memory_length]
        logits = tf.matmul(q, k, transpose_b=True)

        # now add relative logits
        # [batch, num_heads, query_length, max_length]
        rel_logits = tf.layers.dense(q, max_length, name="rel0")
        # [batch, num_heads, query_length, max_length]
        rel_logits = tf.slice(rel_logits, [0, 0, 0, max_length - length],
                              [-1, -1, -1, -1])
        rel_logits = _relative_position_to_absolute_position_masked(rel_logits)
        logits += rel_logits

        if bias is not None:
            logits += bias
        weights = tf.nn.softmax(logits, name="attention_weights")
        # dropping out the attention links for each of the heads
        weights = dropout_with_broadcast_dims(
            weights, 1.0 - dropout_rate, broadcast_dims=dropout_broadcast_dims)
        ret = tf.matmul(weights, v)
        # [batch, num_heads, query_length, memory_length]
        relative_weights = _absolute_position_to_relative_position_masked(
            weights)
        # [batch, num_heads, query_length, memory_length]
        relative_weights = tf.pad(
            relative_weights, [[0, 0], [0, 0], [0, 0], [max_length - length, 0]])
        relative_weights.set_shape([None, None, None, max_length])
        depth_v = shape_list(v)[3]
        ret += tf.layers.dense(relative_weights, depth_v, name="rel1")
        return ret


BatchInfo = collections.namedtuple("BatchInfo", "coordinates, order")


@add_name_scope()
def sparse_dot_product_attention(q, k, v, bi, use_map_fn, experts_params):
    """Sparse multihead self attention.

    Perform an approximation of the full multihead attention by dispatching
    the tokens using their keys/values. Thus the attention matrix are only
    computed each times on a subset of the tokens.

    Notes:
        * The funciton don't perform scaling here (multihead_attention does
        the /sqrt(depth)).
        * The padding should have been removed (so batch size should be 1 but length
        contains the elements from al different batches)
        * Right now, only self attention is supported so length_q and length_kv
        should be identical and the function will add triangular mask.
        * If bi.order is not None, The bias is added inside this function to
        prevent attention to the future.

     Args:
        q (tf.Tensor): Queries of shape [batch, heads, length_q, depth_k]
        k (tf.Tensor): Keys of shape [batch, heads, length_q, depth_k]
        v (tf.Tensor): Values of shape [batch, heads, length_kv, depth_v]
        bi (BatchInfo): Contains the batch coordinates and sequence order
        use_map_fn (bool): Use either tf.map_fn of python for loop to compute the
        heads separately
        experts_params (dict): Additional params for the local expert

    Returns:
        tf.Tensor: Approximation of Softmax(Q.K) * V, of shape
          [batch, heads, length_q, depth_v]
    """
    batch_size, nb_heads, _, depth = shape_list(q)

    @add_name_scope()
    def flatten_first_dims(x):
        """Reshape such that x is [num_heads, -1, depth]."""
        # Case 1: Either constant batch size of size 1 or batch already flattened
        if x.get_shape().as_list()[0] == 1:
            return tf.squeeze(x, axis=0)

        # Case 2: Flatten batch dimension
        x = tf.transpose(x, perm=[1, 0, 2, 3])
        x = tf.reshape(x, [nb_heads, -1, depth])
        return x

    def flatten_batch(x):
        if x is None:
            return x
        return flatten_all_but_last(x)

    q = flatten_first_dims(q)
    k = flatten_first_dims(k)
    v = flatten_first_dims(v)
    bi = BatchInfo(
        coordinates=flatten_batch(bi.coordinates),
        order=flatten_batch(bi.order))

    # Unstack heads
    list_q = tf.unstack(q)  # list[tf.Tensor(shape=batch * length, depth)]
    list_k = tf.unstack(k)
    list_v = tf.unstack(v)

    list_gates_q = []
    list_gates_k = []

    total_loss = 0.0
    # There might be a more optimized way to compute all heads at once
    for single_q, single_k, _ in zip(list_q, list_k, list_v):
        # Each head get its own dispatcher
        lsh_gating = LshGating(
            depth=single_k.get_shape().as_list()[-1], **experts_params)

        list_gates_q.append(lsh_gating.get_gates(single_q))
        list_gates_k.append(lsh_gating.get_gates(single_k))

    gates_q = tf.stack(list_gates_q)
    gates_k = tf.stack(list_gates_k)

    # Process each head separately.
    v_out = map_fn_switch(
        lambda args: dot_product_single_head(bi=bi, *args),
        elems=(q, k, v, gates_q, gates_k),
        dtype=(tf.float32),
        parallel_iterations=2,
        use_map_fn=use_map_fn)

    # Restore original shape as expected by multihead_attention
    if isinstance(batch_size, int) and batch_size == 1:
        v_out = tf.expand_dims(v_out, axis=0)  # Restore batch_size = 1
    else:
        v_out = tf.reshape(v_out, [nb_heads, batch_size, -1, depth])
        v_out = tf.transpose(v_out, [1, 0, 2, 3])
    return v_out, total_loss / nb_heads

multihead_attention_sparse_dot_prod = functools.partial(
    multihead_attention, attention_type=sparse_dot_product_attention)


def masked_within_block_local_attention_1d(q, k, v, block_length=64, name=None):
    """Attention to the source and a neighborhood to the left within a block.

    The sequence is divided into blocks of length block_size.
    Attention for a given query position can only see memory positions
    less than or equal to the query position in the corresponding block

    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      block_length: an integer
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    with tf.variable_scope(
            name, default_name="within_local_attention_1d", values=[q, k, v]):
        v_shape = v.get_shape()
        batch, heads, length, _ = shape_list(q)
        if isinstance(block_length, tf.Tensor):
            const = tf.contrib.util.constant_value(block_length)
            if const is not None:
                block_length = int(const)

        depth_k = shape_list(k)[3]
        depth_v = shape_list(v)[3]
        original_length = length
        padding_size = tf.mod(-length, block_length)
        length += padding_size
        padding = [[0, 0], [0, 0], [0, padding_size], [0, 0]]
        q = tf.pad(q, padding)
        k = tf.pad(k, padding)
        v = tf.pad(v, padding)
        num_blocks = tf.div(length, block_length)
        # compute attention for all subsequent query blocks.
        q = tf.reshape(q, [batch, heads, num_blocks, block_length, depth_k])
        k = tf.reshape(k, [batch, heads, num_blocks, block_length, depth_k])
        v = tf.reshape(v, [batch, heads, num_blocks, block_length, depth_v])
        # attention shape: [batch, heads, num_blocks, block_length, block_length]
        attention = tf.matmul(q, k, transpose_b=True)
        attention += tf.reshape(
            attention_bias_lower_triangle(block_length),
            [1, 1, 1, block_length, block_length])
        attention = tf.nn.softmax(attention)
        # initial output shape: [batch, heads, num_blocks, block_length, depth_v]
        output = tf.matmul(attention, v)
        output = tf.reshape(output, [batch, heads, -1, depth_v])
        output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
        output.set_shape(v_shape)
        return output


def local_attention_1d(q, k, v, block_length=128, filter_width=100, name=None):
    """strided block local self-attention.

    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      block_length: an integer
      filter_width: an integer indicating how much to look left.
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    with tf.variable_scope(
            name, default_name="local_self_attention_1d", values=[q, k, v]):
        v_shape = v.get_shape()
        depth_v = shape_list(v)[3]
        batch_size = shape_list(q)[0]
        num_heads = shape_list(q)[1]
        original_length = shape_list(q)[2]

        # making sure q is a multiple of d
        def pad_to_multiple(x, pad_length):
            x_length = shape_list(x)[2]
            return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

        def pad_l_and_r(x, pad_length):
            return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

        q = pad_to_multiple(q, block_length)
        k = pad_to_multiple(k, block_length)
        v = pad_to_multiple(v, block_length)

        # Setting up q blocks
        new_q_shape = shape_list(q)
        # Setting up q blocks
        q = tf.reshape(q, [
            new_q_shape[0], new_q_shape[1], new_q_shape[2] // block_length,
            block_length, new_q_shape[3]
        ])

        # Setting up k and v values
        k = pad_l_and_r(k, filter_width)
        v = pad_l_and_r(v, filter_width)

        length = shape_list(k)[2]
        full_filter_width = block_length + 2 * filter_width
        # getting gather indices
        indices = tf.range(0, length, delta=1, name="index_range")
        # making indices [1, length, 1] to appy convs
        indices = tf.reshape(indices, [1, -1, 1])
        kernel = tf.expand_dims(tf.eye(full_filter_width), axis=1)
        gather_indices = tf.nn.conv1d(
            tf.cast(indices, tf.float32),
            kernel,
            block_length,
            padding="VALID",
            name="gather_conv")

        gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

        # [length, batch, heads, dim]
        k_t = tf.transpose(k, [2, 0, 1, 3])
        k_new = tf.gather(k_t, gather_indices)

        # [batch, heads, blocks, block_length, dim]
        k_new = tf.transpose(k_new, [2, 3, 0, 1, 4])

        attention_bias = tf.expand_dims(
            embedding_to_padding(k_new) * -1e9, axis=-2)

        v_t = tf.transpose(v, [2, 0, 1, 3])
        v_new = tf.gather(v_t, gather_indices)
        v_new = tf.transpose(v_new, [2, 3, 0, 1, 4])

        output = dot_product_attention(
            q,
            k_new,
            v_new,
            attention_bias,
            dropout_rate=0.,
            name="local_1d")
        output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
        # Remove the padding if introduced
        output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
        output.set_shape(v_shape)
        return output


def masked_local_attention_1d(q,
                              k,
                              v,
                              block_length=128,
                              name=None):
    """Attention to the source position and a neighborhood to the left of it.

    The sequence is divided into blocks of length block_size.
    Attention for a given query position can only see memory positions
    less than or equal to the query position, in the corresponding block
    and the previous block.

    If mask_right is True, then a target position cannot see greater source
    positions.

    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      block_length: an integer
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    with tf.variable_scope(
            name, default_name="local_attention_1d", values=[q, k, v]):
        batch = shape_list(q)[0]
        heads = shape_list(q)[1]
        length = shape_list(q)[2]
        if isinstance(block_length, tf.Tensor):
            const = tf.contrib.util.constant_value(block_length)
            if const is not None:
                block_length = int(const)

        # If (length < 2 * block_length), then we use only one block.
        if isinstance(length, int) and isinstance(block_length, int):
            block_length = length if length < block_length * 2 else block_length
        else:
            block_length = tf.where(
                tf.less(length, block_length * 2), length, block_length)
        depth_k = shape_list(k)[3]
        depth_v = shape_list(v)[3]
        original_length = length
        padding_size = tf.mod(-length, block_length)
        length += padding_size
        padding = [[0, 0], [0, 0], [0, padding_size], [0, 0]]
        q = tf.pad(q, padding)
        k = tf.pad(k, padding)
        v = tf.pad(v, padding)

        if isinstance(length, int) and isinstance(block_length, int):
            num_blocks = length // block_length
        else:
            num_blocks = tf.div(length, block_length)

        # compute attention for the first query block.
        first_q = tf.slice(q, [0, 0, 0, 0], [-1, -1, block_length, -1])
        first_k = tf.slice(k, [0, 0, 0, 0], [-1, -1, block_length, -1])
        first_v = tf.slice(v, [0, 0, 0, 0], [-1, -1, block_length, -1])
        first_output = dot_product_attention(
            first_q,
            first_k,
            first_v,
            attention_bias_lower_triangle(block_length),
            name="fist_block")

        # compute attention for all subsequent query blocks.
        q = tf.reshape(q, [batch, heads, num_blocks, block_length, depth_k])
        k = tf.reshape(k, [batch, heads, num_blocks, block_length, depth_k])
        v = tf.reshape(v, [batch, heads, num_blocks, block_length, depth_v])

        def local(x, depth):
            """Create a local version of the keys or values."""
            prev_block = tf.slice(x, [0, 0, 0, 0, 0],
                                  [-1, -1, num_blocks - 1, -1, -1])
            cur_block = tf.slice(x, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
            local_block = tf.concat([prev_block, cur_block], 3)
            return tf.reshape(local_block,
                              [batch, heads, num_blocks - 1, block_length * 2, depth])

        local_k = local(k, depth_k)
        local_v = local(v, depth_v)
        tail_q = tf.slice(q, [0, 0, 1, 0, 0], [-1, -1, -1, -1, -1])
        tail_q = tf.reshape(tail_q,
                            [batch, heads, num_blocks - 1, block_length, depth_k])
        local_length = shape_list(local_k)[3]

        # [batch, heads, num_blocks - 1, block_length, local_length]
        attention = tf.matmul(tail_q, local_k, transpose_b=True)

        # make sure source_pos <= target_pos
        good_part = ones_matrix_band_part(block_length, local_length,
                                          -1, block_length)
        mask = (1.0 - good_part) * -1e9
        mask = cast_like(mask, attention)
        attention += tf.reshape(mask, [1, 1, 1, block_length, local_length])
        attention = tf.nn.softmax(attention)
        # TODO(noam): figure out how to show a summary for the remaining blocks.
        # The naive way currently causes errors due to empty tensors.
        # output: [batch, heads, num_blocks-1, block_length, depth_v]
        output = tf.matmul(attention, local_v)
        output = tf.reshape(
            output, [batch, heads, (num_blocks - 1) * block_length, depth_v])
        output = tf.concat([first_output, output], axis=2)
        output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
        output = tf.reshape(output, [batch, heads, original_length, depth_v])
        return output


def masked_dilated_self_attention_1d(q,
                                     k,
                                     v,
                                     query_block_size=64,
                                     memory_block_size=64,
                                     gap_size=2,
                                     num_memory_blocks=2,
                                     name=None):
    """dilated self-attention. TODO(avaswani): Try it and write a paper on it.

    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      query_block_size: an integer
      memory_block_size: an integer indicating how much to look left.
      gap_size: an integer indicating the gap size
      num_memory_blocks: how many memory blocks to look at to the left. Each will
        be separated by gap_size.
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    with tf.variable_scope(
            name, default_name="masked_dilated_self_attention_1d", values=[q, k, v]):
        v_list_shape = v.get_shape().as_list()
        v_shape = shape_list(v)
        depth_v = v_shape[3]
        batch_size = v_shape[0]
        num_heads = v_shape[1]
        original_length = shape_list(q)[2]

        # making sure q is a multiple of query block size
        def pad_to_multiple(x, pad_length):
            x_length = shape_list(x)[2]
            return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

        def pad_l(x, left_pad_length):
            return tf.pad(x, [[0, 0], [0, 0], [left_pad_length, 0], [0, 0]])

        q = pad_to_multiple(q, query_block_size)
        v = pad_to_multiple(v, query_block_size)
        k = pad_to_multiple(k, query_block_size)
        q.set_shape(v_list_shape)
        v.set_shape(v_list_shape)
        k.set_shape(v_list_shape)
        # Setting up q blocks
        new_q_shape = shape_list(q)

        # Setting up q blocks
        q = reshape_by_blocks(q, new_q_shape, query_block_size)
        self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
        self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)
        # Setting up k and v windows
        k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
        k = pad_l(k, k_v_padding)
        v = pad_l(v, k_v_padding)
        # Getting gather indices
        index_length = (new_q_shape[2] - query_block_size + memory_block_size)

        indices = tf.range(0, index_length, delta=1, name="index_range")
        # Making indices [1, length, 1] to appy convs
        indices = tf.reshape(indices, [1, -1, 1])
        kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
        gather_indices = tf.nn.conv1d(
            tf.cast(indices, tf.float32),
            kernel,
            query_block_size,
            padding="VALID",
            name="gather_conv")
        gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

        # Get left and right memory blocks for each query
        # [length, batch, heads, dim]
        k_t = tf.transpose(k, [2, 0, 1, 3])
        v_t = tf.transpose(v, [2, 0, 1, 3])

        k_unmasked_windows = gather_dilated_memory_blocks(
            k_t, num_memory_blocks, gap_size, query_block_size, memory_block_size,
            gather_indices)
        v_unmasked_windows = gather_dilated_memory_blocks(
            v_t, num_memory_blocks, gap_size, query_block_size, memory_block_size,
            gather_indices)

        # combine memory windows
        block_q_shape = shape_list(q)
        masked_attention_bias = tf.tile(
            tf.expand_dims(attention_bias_lower_triangle(
                query_block_size), axis=0),
            [block_q_shape[0], block_q_shape[1], block_q_shape[2], 1, 1])
        padding_attention_bias = tf.expand_dims(
            embedding_to_padding(k_unmasked_windows) * -1e9, axis=-2)
        padding_attention_bias = tf.tile(padding_attention_bias,
                                         [1, 1, 1, query_block_size, 1])
        attention_bias = tf.concat(
            [masked_attention_bias, padding_attention_bias], axis=-1)
        # Combine memory windows
        k_windows = tf.concat([self_k_part, k_unmasked_windows], 3)
        v_windows = tf.concat([self_v_part, v_unmasked_windows], 3)
        output = dot_product_attention(
            q,
            k_windows,
            v_windows,
            attention_bias,
            dropout_rate=0.,
            name="dilated_1d")
        output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
        # Remove the padding if introduced
        output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
        output.set_shape(v_list_shape)
        return output


def dilated_self_attention_1d(q,
                              k,
                              v,
                              query_block_size=128,
                              memory_block_size=128,
                              gap_size=2,
                              num_memory_blocks=2,
                              name=None):
    """dilated self-attention.

    Args:
      q: a Tensor with shape [batch, heads, length, depth_k]
      k: a Tensor with shape [batch, heads, length, depth_k]
      v: a Tensor with shape [batch, heads, length, depth_v]
      query_block_size: an integer indicating size of query block
      memory_block_size: an integer indicating the size of a memory block.
      gap_size: an integer indicating the gap size
      num_memory_blocks: how many memory blocks to look at to the left and right.
        Each will be separated by gap_size.
      name: an optional string

    Returns:
      a Tensor of shape [batch, heads, length, depth_v]
    """
    with tf.variable_scope(
            name, default_name="dilated_self_attention_1d", values=[q, k, v]):
        v_list_shape = v.get_shape().as_list()
        v_shape = shape_list(v)
        depth_v = v_shape[3]
        batch_size = v_shape[0]
        num_heads = v_shape[1]
        original_length = shape_list(q)[2]

        # making sure q is a multiple of query block size
        def pad_to_multiple(x, pad_length):
            x_length = shape_list(x)[2]
            return tf.pad(x, [[0, 0], [0, 0], [0, -x_length % pad_length], [0, 0]])

        def pad_l_and_r(x, pad_length):
            return tf.pad(x, [[0, 0], [0, 0], [pad_length, pad_length], [0, 0]])

        q = pad_to_multiple(q, query_block_size)
        v = pad_to_multiple(v, query_block_size)
        k = pad_to_multiple(k, query_block_size)

        q.set_shape(v_list_shape)
        v.set_shape(v_list_shape)
        k.set_shape(v_list_shape)
        # Setting up q blocks
        new_q_shape = shape_list(q)
        # Setting up q blocks
        q = reshape_by_blocks(q, new_q_shape, query_block_size)
        self_k_part = reshape_by_blocks(k, new_q_shape, query_block_size)
        self_v_part = reshape_by_blocks(v, new_q_shape, query_block_size)

        # Setting up k and v windows
        k_v_padding = (gap_size + memory_block_size) * num_memory_blocks
        k = pad_l_and_r(k, k_v_padding)
        v = pad_l_and_r(v, k_v_padding)
        # getting gather indices
        index_length = (new_q_shape[2] - query_block_size + memory_block_size)
        indices = tf.range(0, index_length, delta=1, name="index_range")
        # making indices [1, length, 1] to appy convs
        indices = tf.reshape(indices, [1, -1, 1])
        kernel = tf.expand_dims(tf.eye(memory_block_size), axis=1)
        gather_indices = tf.nn.conv1d(
            tf.cast(indices, tf.float32),
            kernel,
            query_block_size,
            padding="VALID",
            name="gather_conv")

        gather_indices = tf.squeeze(tf.cast(gather_indices, tf.int32), axis=0)

        # get left and right memory blocks for each query
        # [length, batch, heads, dim]
        k_t = tf.transpose(k, [2, 0, 1, 3])
        v_t = tf.transpose(v, [2, 0, 1, 3])
        left_k = gather_dilated_memory_blocks(
            k_t[:-k_v_padding, :, :, :], num_memory_blocks, gap_size,
            query_block_size, memory_block_size, gather_indices)
        left_v = gather_dilated_memory_blocks(
            v_t[:-k_v_padding, :, :, :], num_memory_blocks, gap_size,
            query_block_size, memory_block_size, gather_indices)

        right_k = gather_dilated_memory_blocks(
            k_t[k_v_padding:, :, :, :],
            num_memory_blocks,
            gap_size,
            query_block_size,
            memory_block_size,
            gather_indices,
            direction="right")
        right_v = gather_dilated_memory_blocks(
            v_t[k_v_padding:, :, :, :],
            num_memory_blocks,
            gap_size,
            query_block_size,
            memory_block_size,
            gather_indices,
            direction="right")

        k_windows = tf.concat([left_k, self_k_part, right_k], axis=3)
        v_windows = tf.concat([left_v, self_v_part, right_v], axis=3)
        attention_bias = tf.expand_dims(
            embedding_to_padding(k_windows) * -1e9, axis=-2)

        output = dot_product_attention(
            q,
            k_windows,
            v_windows,
            attention_bias,
            dropout_rate=0.,
            name="dilated_1d")
        output = tf.reshape(output, [batch_size, num_heads, -1, depth_v])
        # Remove the padding if introduced
        output = tf.slice(output, [0, 0, 0, 0], [-1, -1, original_length, -1])
        output.set_shape(v_list_shape)
        return output
