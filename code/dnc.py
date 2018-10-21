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
"""DNC Cores.

These modules create a DNC core. They take input, pass parameters to the memory
access module, and integrate the output of memory to form an output.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy as np
import sonnet as snt
import tensorflow as tf

from dnc import access
from dnc import util

from tensor2tensor.layers import common_attention
from tensor2tensor.layers import common_layers
from tensor2tensor.models import transformer
from tensor2tensor.utils import expert_utils
from tensor2tensor.models.research import universal_transformer_util
#from tensor2tensor.models.research import universal_transformer


DNCState = collections.namedtuple('DNCState', ('access_output', 'access_state',
                                               'controller_state'))


class DNCTransformer(snt.RNNCore, transformer.Transformer):
    """DNC core module.

    Contains controller and memory access module.
    """

    def __init__(self,
                 features,
                 access_config,
                 controller_config,
                 output_size,
                 clip_value=None,
                 name='dnc'):
        """Initializes the DNC core.

        Args:
          access_config: dictionary of access module configurations.
          controller_config: dictionary of controller (LSTM) module configurations.
          output_size: output dimension size of core.
          clip_value: clips controller and core output values to between
              `[-clip_value, clip_value]` if specified.
          name: module name (default 'dnc').

        Raises:
          TypeError: if direct_input_size is not None for any access module other
            than KeyValueMemory.
        """
        super(DNCTransformer, self).__init__(name=name)
        
        with self._enter_variable_scope():        
            self._access = access.MemoryAccess(**access_config)
            self._controller = self.body(
                **controller_config)

        self._access_output_size = np.prod(self._access.output_size.as_list())
        self._output_size = output_size
        self._clip_value = clip_value or 0

        self._output_size = tf.TensorShape([output_size])
        self._state_size = DNCState(
            access_output=self._access_output_size,
            access_state=self._access.state_size,
            controller_state=self._controller())

    def encode(self, inputs, target_space, hparams, features=None, losses=None):
        """Encode Universal Transformer inputs.

        It is similar to "transformer.encode", but it uses
        "universal_transformer_util.universal_transformer_encoder" instead of
        "transformer.transformer_encoder".

        Args:
        inputs: Transformer inputs [batch_size, input_length, input_height,
            hidden_dim] which will be flattened along the two spatial dimensions.
        target_space: scalar, target space ID.
        hparams: hyperparmeters for model.
        features: optionally pass the entire features dictionary as well.
            This is needed now for "packed" datasets.
        losses: Unused.

        Returns:
        Tuple of:
            encoder_output: Encoder representation.
                [batch_size, input_length, hidden_dim]
            encoder_decoder_attention_bias: Bias and mask weights for
                encoder-decoder attention. [batch_size, input_length]
            encoder_extra_output: which is extra encoder output used in some
                variants of the model (e.g. in ACT, to pass the ponder-time to body)
        """
        del losses

        inputs = common_layers.flatten4d3d(inputs)

        encoder_input, self_attention_bias, encoder_decoder_attention_bias = (
            transformer.transformer_prepare_encoder(
                inputs, target_space, hparams, features=features))

        encoder_input = tf.nn.dropout(encoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)

        (encoder_output, encoder_extra_output) = (
            universal_transformer_util.universal_transformer_encoder(
                encoder_input,
                self_attention_bias,
                hparams,
                nonpadding=transformer.features_to_nonpadding(
                    features, "inputs"),
                save_weights_to=self.attention_weights))

        return encoder_output, encoder_decoder_attention_bias, encoder_extra_output

    def decode(self, decoder_input, encoder_output, encoder_decoder_attention_bias, decoder_self_attention_bias,
               hparams, cache=None, nonpadding=None, losses=None):
        """Decode Universal Transformer outputs from encoder representation.

        It is similar to "transformer.decode", but it uses
        "universal_transformer_util.universal_transformer_decoder" instead of
        "transformer.transformer_decoder".

        Args:
        decoder_input: inputs to bottom of the model. [batch_size, decoder_length,
            hidden_dim]
        encoder_output: Encoder representation. [batch_size, input_length,
            hidden_dim]
        encoder_decoder_attention_bias: Bias and mask weights for encoder-decoder
            attention. [batch_size, input_length]
        decoder_self_attention_bias: Bias and mask weights for decoder
            self-attention. [batch_size, decoder_length]
        hparams: hyperparmeters for model.
        cache: Unimplemented.
        nonpadding: optional Tensor with shape [batch_size, decoder_length]
        losses: Unused.

        Returns:
        Tuple of:
            Final decoder representation. [batch_size, decoder_length,
                hidden_dim]
            encoder_extra_output: which is extra encoder output used in some
                variants of the model (e.g. in ACT, to pass the ponder-time to body)

        """
        del losses
        del cache

        decoder_input = tf.nn.dropout(decoder_input,
                                      1.0 - hparams.layer_prepostprocess_dropout)

        (decoder_output, dec_extra_output) = (
            universal_transformer_util.universal_transformer_decoder(
                decoder_input,
                encoder_output,
                decoder_self_attention_bias,
                encoder_decoder_attention_bias,
                hparams,
                nonpadding=nonpadding,
                save_weights_to=self.attention_weights))

        return tf.expand_dims(decoder_output, axis=2), dec_extra_output

    def body(self, features):
        """Universal Transformer main model_fn.

        Args:
        features: Map of features to the model. Should contain the following:
            "inputs": Transformer inputs [batch_size, input_length, hidden_dim]
            "targets": Target decoder outputs.
                [batch_size, decoder_length, hidden_dim]
            "target_space_id"

        Returns:
        Final decoder representation. [batch_size, decoder_length, hidden_dim]
        """
        hparams = self._hparams
        if hparams.add_postion_timing_signal:
            # Turning off addition of positional embedding in the encoder/decoder
            # preparation as we do it in the beginning of each step.
            hparams.pos = None

        if self.has_input:
            inputs = features["inputs"]
            target_space = features["target_space_id"]
            (encoder_output, encoder_decoder_attention_bias,
             enc_extra_output) = self.encode(
                inputs, target_space, hparams, features=features)
        else:
            (encoder_output, encoder_decoder_attention_bias,
             enc_extra_output) = (None, None, (None, None))

        targets = features["targets"]
        targets = common_layers.flatten4d3d(targets)

        (decoder_input,
         decoder_self_attention_bias) = transformer.transformer_prepare_decoder(
            targets, hparams, features=features)

        decoder_output, dec_extra_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=transformer.features_to_nonpadding(features, "targets"))

        expected_attentions = features.get("expected_attentions")
        if expected_attentions is not None:
            print('returning attention loss')
            attention_loss = common_attention.encoder_decoder_attention_loss(
                expected_attentions, self.attention_weights,
                hparams.expected_attention_loss_type,
                hparams.expected_attention_loss_multiplier)
            return decoder_output, {"attention_loss": attention_loss}

        if hparams.recurrence_type == "act" and hparams.act_loss_weight != 0:
            print('returning act loss')
            if self.has_input:
                enc_ponder_times, enc_remainders = enc_extra_output
                enc_act_loss = (
                    hparams.act_loss_weight *
                    tf.reduce_mean(enc_ponder_times + enc_remainders))
            else:
                enc_act_loss = 0.0

            (dec_ponder_times, dec_remainders) = dec_extra_output
            dec_act_loss = (
                hparams.act_loss_weight *
                tf.reduce_mean(dec_ponder_times + dec_remainders))
            act_loss = enc_act_loss + dec_act_loss
            tf.contrib.summary.scalar("act_loss", act_loss)
            return decoder_output, {"act_loss": act_loss}

        #grads = get_grads_and_vars(attention_loss)
        # dec_out_and_grads = tf.concat([decoder_output, grads], 1)  # ¿0 or 1?
        access_output, access_state = self._access(decoder_output,
                                                   dec_extra_output)

        return decoder_output, DNCState(
            access_output=access_output,
            access_state=access_state,
            controller_state=dec_extra_output)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._output_size

    def get_grads_and_vars(self, loss):
        trainable_vars = tf.trainable_variables()
        grads = tf.gradients(loss, trainable_vars)
        return grads
