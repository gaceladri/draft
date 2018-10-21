# Copyright 2017 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Example script to train the DNC on a repeated copy task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import sys
import os
import random
import numpy as np

from dnc import dnc
from dnc.load_data import *
from dnc.util import positional_encoding, multihead_attention, feedforward, label_smoothing, normalize
from dnc import hyperparameters as hp

import tensorflow as tf
import sonnet as snt


class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.x, self.y, self.num_batch = get_batch_data()  # (N, T)
            else:
                self.x = tf.placeholder(tf.float32, shape=(None, hp.maxlen))
                self.y = tf.placeholder(tf.float32, shape=(None, hp.maxlen))

            # define decoder inputs
            self.decoder_inputs = tf.concat(
                (tf.ones_like(self.y[:, :1])*2, self.y[:, :-1]), -1)

            # Encoder
            with tf.variable_scope("encoder"):
                # Embedding
                self.enc = embedding(self.x,
                                     vocab_size=len(),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="enc_embed")

                # Positional encoding
                if hp.sinusoid:
                    self.enc += positional_encoding(self.x,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="enc_pe")

                # Dropout
                self.enc = tf.layers.dropout(self.enc,
                                             rate=hp.dropout_rate,
                                             training=tf.convert_to_tensor(is_training))

                # Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # Multihead attention
                        self.enc = multihead_attention(queries=self.enc,
                                                       keys=self.enc,
                                                       num_units=hp.hidden_units,
                                                       num_heads=hp.num_heads,
                                                       dropout_rate=hp.dropout_rate,
                                                       is_training=is_training,
                                                       causality=False)
                        self.enc = feedforward(self.enc, num_units=[
                                               4*hp.hidden_units, hp.hidden_units])

            # Decoder
            with tf.variable_scope("decoder"):
                # Embedding
                self.dec = embedding(self.decoder_inputs,
                                     vocab_size=len(),
                                     num_units=hp.hidden_units,
                                     scale=True,
                                     scope="dec_embed")

                # Positional encoding
                if hp.sinusoid:
                    self.dec += positional_encoding(self.decoder_inputs,
                                                    vocab_size=hp.maxlen,
                                                    num_units=hp.hidden_units,
                                                    zero_pad=False,
                                                    scale=False,
                                                    scope="dec_pe")
                else:
                    self.dec += embedding(tf.tile(tf.expand_dims(tf.range(tf.shape(self.decoder_inputs)[1]), 0), [tf.shape(self.decoder_inputs)[0], 1]),
                                          vocab_size=hp.maxlen,
                                          num_units=hp.hidden_units,
                                          zero_pad=False,
                                          scale=False,
                                          scope="dec_pe")

                # Dropout
                self.dec = tf.layers.dropout(self.dec,
                                            rate=hp.dropout_rate,
                                            training=tf.convert_to_tensor(is_training))
                
                # Blocks
                for i in range(hp.num_blocks):
                    with tf.variable_scope("num_blocks_{}".format(i)):
                        # Multihead attention (self-attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                        keys=self.dec,
                                                        num_units=hp.hidden_units,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        causality=True,
                                                        scope="self-attention")

                        # Multihead attention (vanilla attention)
                        self.dec = multihead_attention(queries=self.dec,
                                                        keys=self.enc,
                                                        num_units=hp.hidden_units,
                                                        num_heads=hp.num_heads,
                                                        dropout_rate=hp.dropout_rate,
                                                        is_training=is_training,
                                                        causality=False,
                                                        scope="vanilla_attention")

                        # Feed forward
                        self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])

            # Final linear projection
            self.logits = tf.layers.dense(self.dec, len(en2idx))
            self.preds = tf.to_float(tf.arg_max(self.logits, dimension=-1))
            self.istarget = tf.to_float(tf.not_equal(self.y, 0))
            self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.y))*self.istarget) / (tf.reduce_sum(self.istarget))
            tf.summary.scalar('acc', self.acc)

            if is_training:
                # Loss
                self.y_smoothed = label_smoothing(tf.one_hot(self.y, depth=len()))
                self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.y_smoothed)
                self.mean_loss = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))

                # Training scheme 
                self.global_step = tf.Variable(0, name="global_step", trainable=False)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=hp.lr, beta1=0.9, beta2=0.98, epsilon=1e-8)
                self.train_op = self.optimizer.minimize(self.mean_loss, global_step=self.global_step)

                # Summary
                tf.summary.scalar("mean_loss", self.mean_loss)
                self.merged = tf.summary.merge_all()

                


def run_model(input_sequence, output_size):
    """Runs model on input sequence."""

    access_config = {
        "memory_size": FLAGS.memory_size,
        "word_size": FLAGS.word_size,
        "num_reads": FLAGS.num_read_heads,
        "num_writes": FLAGS.num_write_heads,
    }
    controller_config = {
        "hidden_size": FLAGS.hidden_size,
    }
    clip_value = FLAGS.clip_value

    dnc_core = dnc.DNCTransformer(
        access_config, controller_config, output_size, clip_value)
    initial_state = dnc_core.initial_state(FLAGS.batch_size)
    output_sequence, _ = tf.nn.dynamic_rnn(
        cell=dnc_core,
        inputs=input_sequence,
        time_major=True,
        initial_state=initial_state)

    return output_sequence


def train(num_training_iterations, report_interval):
    """Trains the DNC and periodically reports the loss."""

    dataset = repeat_copy.RepeatCopy(FLAGS.num_bits, FLAGS.batch_size,
                                     FLAGS.min_length, FLAGS.max_length,
                                     FLAGS.min_repeats, FLAGS.max_repeats)
    dataset_tensors = dataset()

    output_logits = run_model(
        dataset_tensors.observations, dataset.target_size)
    # Used for visualization.
    output = tf.round(
        tf.expand_dims(dataset_tensors.mask, -1) * tf.sigmoid(output_logits))

    train_loss = dataset.cost(output_logits, dataset_tensors.target,
                              dataset_tensors.mask)

    # Set up optimizer with global norm clipping.
    trainable_variables = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(
        tf.gradients(train_loss, trainable_variables), FLAGS.max_grad_norm)

    global_step = tf.get_variable(
        name="global_step",
        shape=[],
        dtype=tf.int64,
        initializer=tf.zeros_initializer(),
        trainable=False,
        collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])

    optimizer = tf.train.RMSPropOptimizer(
        FLAGS.learning_rate, epsilon=FLAGS.optimizer_epsilon)
    train_step = optimizer.apply_gradients(
        zip(grads, trainable_variables), global_step=global_step)

    saver = tf.train.Saver()

    if FLAGS.checkpoint_interval > 0:
        hooks = [
            tf.train.CheckpointSaverHook(
                checkpoint_dir=FLAGS.checkpoint_dir,
                save_steps=FLAGS.checkpoint_interval,
                saver=saver)
        ]
    else:
        hooks = []

    # Train.
    with tf.train.SingularMonitoredSession(
            hooks=hooks, checkpoint_dir=FLAGS.checkpoint_dir) as sess:

        start_iteration = sess.run(global_step)
        total_loss = 0

        for train_iteration in range(start_iteration, num_training_iterations):
            _, loss = sess.run([train_step, train_loss])
            total_loss += loss

            if (train_iteration + 1) % report_interval == 0:
                dataset_tensors_np, output_np = sess.run(
                    [dataset_tensors, output])
                dataset_string = dataset.to_human_readable(dataset_tensors_np,
                                                           output_np)
                tf.logging.info("%d: Avg training loss %f.\n%s",
                                train_iteration, total_loss / report_interval,
                                dataset_string)
                total_loss = 0


def main(argv):
    tf.logging.set_verbosity(tf.logging.INFO)  # Print INFO log messages.
    set_random_seed(FLAGS.random_seed)
    usr_dir.import_usr_dir(FLAGS.t2t_usr_dir)
    maybe_log_registry_and_exit()

    if FLAGS.generate_data:
        generate_data()

    if argv:
        set_hparams_from_args(argv[1:])
    hparams = create_hparams()

    exp_fn = create_experiment_fn()
    exp = exp_fn(create_run_config(hparams), hparams)
    if is_chief():
        save_metadata(hparams)
    execute_schedule(exp)

    train(FLAGS.num_training_iterations, FLAGS.report_interval)


if __name__ == "__main__":
    tf.app.run()
