from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import functools
import code.load_data as data_module

from util import attention_layer, get_shape_list, reshape_from_matrix, reshape_to_matrix
from tensor2tensor.utils import expert_utils as eu
from tensor2tensor.layers import common_attention as ca


def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
    """Multi-headed, multi-layer Transformer from "Attention is All You Need".

    This is almost an exact implementation of the original Transformer encoder.

    See the original paper:
    https://arxiv.org/abs/1706.03762

    Also see:
    https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

    Args:
      input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
      attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
        seq_length], with 1 for positions that can be attended to and 0 in
        positions that should not be.
      hidden_size: int. Hidden size of the Transformer.
      num_hidden_layers: int. Number of layers (blocks) in the Transformer.
      num_attention_heads: int. Number of attention heads in the Transformer.
      intermediate_size: int. The size of the "intermediate" (a.k.a., feed
        forward) layer.
      intermediate_act_fn: function. The non-linear activation function to apply
        to the output of the intermediate/feed-forward layer.
      hidden_dropout_prob: float. Dropout probability for the hidden layers.
      attention_probs_dropout_prob: float. Dropout probability of the attention
        probabilities.
      initializer_range: float. Range of the initializer (stddev of truncated
        normal).
      do_return_all_layers: Whether to also return all layers or just the final
        layer.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size], the final
      hidden layer of the Transformer.

    Raises:
      ValueError: A Tensor shape or parameter is invalid.
    """
    if hidden_size % num_attention_heads != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (hidden_size, num_attention_heads))

    attention_head_size = int(hidden_size / num_attention_heads)
    input_shape = get_shape_list(input_tensor, expected_rank=3)
    batch_size = input_shape[0]
    seq_length = input_shape[1]
    input_width = input_shape[2]

    # The Transformer performs sum residuals on all layers so the input needs
    # to be the same as the hidden size.
    if input_width != hidden_size:
        raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                         (input_width, hidden_size))

    # We keep the representation as a 2D tensor to avoid re-shaping it back and
    # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
    # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
    # help the optimizer.
    prev_output = reshape_to_matrix(input_tensor)

    # TODO: after first test that the model runs, change the variable scope to share weigths between
    #  Time-Series (as a transfer learning method)

    # TODO: check the attention bias and check the encodeco_input_bias from original model
    with tf.variable_scope("encoder"):
        all_encoder_outputs = []
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                enc_input = prev_output

                with tf.variable_scope("attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_head = attention_layer(
                            from_tensor=enc_input,
                            to_tensor=enc_input,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length)
                        attention_heads.append(attention_head)

                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        # In the case where we have other sequences, we just concatenate
                        # them to the self-attention head before the projection.
                        attention_output = tf.concat(attention_heads, axis=-1)

                    # Run a linear projection of `hidden_size` then add a residual
                    # with `layer_input`.
                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=create_initializer(initializer_range))
                        attention_output = dropout(
                            attention_output, hidden_dropout_prob)
                        attention_output = layer_norm(
                            attention_output + enc_input)

                # The activation is only applied to the "intermediate" hidden layer.
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(
                        attention_output,
                        intermediate_size,
                        activation=intermediate_act_fn,
                        kernel_initializer=create_initializer(initializer_range))

                # Down-project back to `hidden_size` then add the residual.
                with tf.variable_scope("output"):
                    enc_output = tf.layers.dense(
                        intermediate_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    layer_output_enc = dropout(enc_output, hidden_dropout_prob)
                    layer_output_enc = layer_norm(
                        enc_output + attention_output)
                    prev_output_enc = enc_output
                    all_encoder_outputs.append(layer_output_enc)

    # TODO: check the attention bias and check the deco_input_bias from original model
    with tf.variable_scope("decoder"):
        all_decoder_outputs = []
        for layer_idx in range(num_hidden_layers):
            with tf.variable_scope("layer_%d" % layer_idx):
                deco_input = prev_output

                with tf.variable_scope("attention"):
                    attention_heads = []
                    with tf.variable_scope("self"):
                        attention_head = attention_layer(
                            from_tensor=deco_input,
                            to_tensor=deco_input,
                            attention_mask=attention_mask,
                            num_attention_heads=num_attention_heads,
                            size_per_head=attention_head_size,
                            attention_probs_dropout_prob=attention_probs_dropout_prob,
                            initializer_range=initializer_range,
                            do_return_2d_tensor=True,
                            batch_size=batch_size,
                            from_seq_length=seq_length,
                            to_seq_length=seq_length)
                        attention_heads.append(attention_head)

                    attention_output = None
                    if len(attention_heads) == 1:
                        attention_output = attention_heads[0]
                    else:
                        # In the case where we have other sequences, we just concatenate
                        # them to the self-attention head before the projection.
                        attention_output = tf.concat(attention_heads, axis=-1)

                    # Run a linear projection of `hidden_size` then add a residual
                    # with `layer_input`.
                    with tf.variable_scope("output"):
                        attention_output = tf.layers.dense(
                            attention_output,
                            hidden_size,
                            kernel_initializer=create_initializer(initializer_range))
                        attention_output = dropout(
                            attention_output, hidden_dropout_prob)
                        attention_output = layer_norm(
                            attention_output + deco_input)

                # The activation is only applied to the "intermediate" hidden layer.
                with tf.variable_scope("intermediate"):
                    intermediate_output = tf.layers.dense(
                        attention_output,
                        intermediate_size,
                        activation=intermediate_act_fn,
                        kernel_initializer=create_initializer(initializer_range))

                # Down-project back to `hidden_size` then add the residual.
                with tf.variable_scope("output"):
                    deco_output = tf.layers.dense(
                        intermediate_output,
                        hidden_size,
                        kernel_initializer=create_initializer(initializer_range))
                    deco_output = dropout(deco_output, hidden_dropout_prob)
                    deco_output = layer_norm(deco_output + attention_output)
                    prev_output_deco = deco_output
                    all_decoder_outputs.append(deco_output)

    if do_return_all_layers:
        final_enc_outputs = []
        final_deco_outputs = []
        for layer_output in all_layer_outputs:
            final_encoder = reshape_from_matrix(enc_output, input_shape)
            final_decoder = reshape_from_matrix(deco_output, input_shape)
            final_enc_outputs.append(final_encoder)
            final_deco_outputs.append(final_decoder)
        return final_enc_outputs, final_deco_outputs
    else:
        final_enc_outputs = reshape_from_matrix(prev_output_enc, input_shape)
        final_deco_outputs = reshape_from_matrix(prev_output_deco, input_shape)
        return final_enc_outputs, final_deco_outputs


class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.training_data, self.test_data, self.sub_format, self.pred_windows = data_module.read_data()
            else:
                self.x = tf.placeholder(
                    tf.float32, shape=(None, hparams.maxlen, features))
                self.y = tf.placeholder(
                    tf.float32, shape=(None, hparams.maxlen, features))

            pred_window =
            num_preds =
            num_pred_hours =
            for ser_id, ser_data in self.training_data.groupby('series_id'):
                train_series = self.training_data[self.training_data.series_id == ser_id]
                self._X, self._y, self.scaler = data_module.prepare_training_data(
                    ser_data.consumption, 15)

                # transformer = transformer_model(
                #    self._X, self._y, enc_layers, dec_layers, hparams)
                self.logits = tf.layers.dense(self.dec, num_pred_hours)

    def create_model(config, is_training, input_ids, input_mask, segment_ids,
                     labels, num_labels, num_pred_window=1):
        model = BertModel(
            config=config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask)

        enc_layers, deco_layers = model.get_all_outputs()

        features_size = deco_layers.shape[-1].value

        output_weigths = tf.get_variable(
            "output_weigths", [num_labels, features_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        with tf.variable_scope("loss"):
            if is_training:
                enc_layers = tf.nn.dropout(enc_layers, keep_prob=0.9)
                deco_layers = tf.nn.dropout(deco_layers, keep_prob=0.9)

            logits = tf.matmul(deco_layer, output_weigths, transpose_b=True)
            logits = tf.nn.bias_add(logits, output_bias)
            logits = tf.layers.dense(logits, num_pred_window)

            loss = tf.losses.mean_squared_error(labels, logits)

            return (loss, logits)


def model_fn_builder(config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
    """Returns `model_fn` closure for TPUEstimator."""
    def model_fn(features, labels, mode, params):
        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info(" name = %s, shape = %s" %
                            (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (loss, logits) = create_model(
            config, is_training, input_ids, input_mask, segment_ids, label_ids, num_labels)

        tvars = tf.trainable_variables()

        if init_checkpoint:
            (assigment_map, initialized_variable_names
             ) = get_assigment_map_from_checkpoint(tvars, init_checkpoint)
            tf.train.init_from_checkpoint(init_checkpoint, assigment_map)

        tf.logging.info("*** Trainable Variables ***")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*
            tf.logging.info(" name = %s, shape = %s%s", var.name, var.shape,
                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            global_step = tf.train.get_or_create_global_step()

            learning_rate = tf.constant(
                value=learning_rate, shape=[], dtype=tf.float32)
            # Implements linear decay of the learning rate.
            learning_rate = tf.train_polynomial_decay(
                learning_rate,
                global_step,
                num_train_steps,
                end_learning_rate=0.0,
                power=1.0,
                cycle=False)

            if num_warmup_steps:
                global_steps_int = tf.cast(global_step, tf.int32)
                warmup_steps_int = tf.constant(
                    num_warmup_steps, dtype=tf.int32)

                global_steps_float = tf.cast(global_steps_int, tf.float32)
                warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

                is_warmup = tf.cast(global_steps_int <
                                    warmup_steps_int, tf.float32)
                learning_rate = (
                    (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

            # It is recommended that you use this optimizer for fine tuning, since this
            # is how the model was trained ( note that the Adam m/v variables are NOT
            # loaded from init_checkpoint. )
            optimizer = AdamWeightDecayOptimizer(
                learning_rate=learning_rate,
                weight_decay_rate=0.01,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-6,
                exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

            tvars = tf.trainable_variables()
            grads = tf.gradients(loss, tvars)

            # This is how the model was pre-trained.
            (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

            train_op = optimizer.apply_gradients(
                zip(grads, tvars), global_step=global_step)

            new_global_step = global_step + 1
            train_op = tf.group(
                train_op, [global_step.assign(new_global_step)])

            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                train_op=train_op)

        elif mode == tf.estimator.ModeKeys.EVAL:
            def metric_fn(loss, labels, logits):
                # TODO: check that the output of logits is the desired prediction number
                predictions = logits
                mae = tf.metrics.mean_absolute_error(labels, predictions)
                loss = loss
                return {
                    "eval_mae": mae,
                    "eval_loss": loss
                }  # TODO: add more metrics

            eval_metrics = (metric_fn, [loss, labels, logits])
            output_spec = tf.estimator.EstimatorSpec(
                mode=mode,
                loss=loss,
                eval_metrics=eval_metrics)
        else:
            raise ValueError(
                "Only TRAIN and EVAL modes are supported: %s" % (mode))
        return output_spec

    return model_fn


class BertModel(object):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Example usage:

    ```python

    input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
    input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])

    config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
        num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

    model = modeling.BertModel(config=config, is_training=True,
        input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

    label_embeddings = tf.get_variable(...)
    pooled_output = model.get_pooled_output()
    logits = tf.matmul(pooled_output, label_embeddings)
    ...
    ```
    """

    def __init__(self,
                 config,
                 is_training,
                 input_ids,
                 input_mask=None,
                 token_type_ids=None,
                 scope=None):
        """Constructor for BertModel.

        Args:
        config: `BertConfig` instance.
        is_training: bool. rue for training model, false for eval model. Controls
            whether dropout will be applied.
        input_ids: int32 Tensor of shape [batch_size, seq_length].
        input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
        token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
        scope: (optional) variable scope. Defaults to "bert".

        Raises:
        ValueError: The config is invalid or one of the input tensor shapes
            is invalid.
        """
        config = copy.deepcopy(config)
        if not is_training:
            config.hidden_dropout_prob = 0.0
            config.attention_probs_dropout_prob = 0.0

        input_shape = get_shape_list(input_ids, expected_rank=3)
        batch_size = input_shape[0]
        seq_length = input_shape[1]
        features_size = input_shape[2]

        # TODO: add positional timing signal. is that necesary when dealing with Time-series?
        if input_mask is None:
            input_mask = tf.ones(
                shape=[batch_size, seq_length], dtype=tf.int32)  # TODO: check the 3rd dimension

        if token_type_ids is None:
            token_type_ids = tf.zeros(
                shape=[batch_size, seq_length], dtype=tf.int32)  # TODO: check the 3rd dimension

        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)  # TODO: adapt the att_mask to Time-series

        self.final_enc_outputs, self.final_deco_outputs = transformer_model(
            input_tensor=self._X,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

    def get_all_outputs(self):
        return self.final_enc_outputs, self.final_deco_outputs


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)

    config =  # Load config from ...

    if FLAGS.do_train:
        train_examples = get_train_examples(FLAGS.data_dir)
        num_train_steps = int(len(train_examples) /
                              FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

    model_fn = model_fn_builder(
        config=config,
        num_labels=len(label),
        init_checkpoint=FLAGS.init_checkpoint,
        learning_rate=FLAGS.learning_rate,
        num_train_steps=num_train_steps,
        num_warmup_steps=num_warmup_steps)

    tpu_cluster_resolver = None
    run_config = tf.contrib.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        master=FLAGS.master,
        model_dir=FLAGS.output_dir,
        save_checkpoints_steps=FLAGS.save_checkpoints_steps,
        tpu_config=tf.contrib.tpu.TPUConfig(
            iterations_per_loop=FLAGS.iterations_per_loop,
            num_shards=FLAGS.num_tpu_cores,
            per_host_input_for_training=is_per_host))

    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        eval_batch_sisze=FLAGS.eval_batch_size)

    if FLAGS.do_train:
        train_file =
        tf.logging.info("*** Running training ***")
        tf.logging.info(" Num examples = %d", len(train_examples))
        tf.logging.info(" Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info(" Num steps = %d", num_train_steps)
        train_input_fn =
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = get_eval_examples()
        tf.logging.info("*** Running evaluation ***")
        tf.logging.info(" Num examples = %d", len(eval_examples))
        tf.logging.info(" Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn =

        result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)
