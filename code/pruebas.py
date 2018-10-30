from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from code.util import transformer_prepare_encoder, transformer_prepare_decoder
from code.util import _extract_layer_types, add_name_scope, add_var_scope, layer_preprocess, layer_postprocess
from code.hyperparameters import hparams
from tensor2tensor.layers.common_attention import get_standardized_layers
from tensor2tensor.layers import common_attention, common_layers

import tensorflow as tf
import code.load_data as data_module

hparams = hparams
enc_layers = hparams.enc_layers
dec_layers = hparams.dec_layers

cache = dict(extra_loss=0.0)


def preprocess(x, hparams):
    return layer_preprocess(x, hparams)


def prepostprocess(fct):
    """Apply processing and capture the extra loss."""
    @add_var_scope()
    def decorated(x, *args, **kwargs):
        x = preprocess(x, hparams)
        y, loss = fct(x, *args, **kwargs)
        cache["extra_loss"] += loss
        return layer_postprocess(x, y, hparams)
    return decorated


layers = get_standardized_layers(hparams)


class Graph():
    def __init__(self, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.training_data, self.test_data, self.sub_format, self.pred_windows = data_module.read_data()
            else:
                self.x = tf.placeholder(
                    tf.float32, shape=(None, hparams.maxlen))
                self.y = tf.placeholder(
                    tf.float32, shape=(None, hparams.maxlen))

            pred_window = 
            num_preds = 
            num_pred_hours = 
            for ser_id, ser_data in self.training_data.groupby('series_id'):
                train_series = self.training_data[self.training_data.series_id == ser_id]
                self._X, self._y, self.scaler = data_module.prepare_training_data(
                    ser_data.consumption, 15)

                transformer = transformer(
                    self._X, self._y, enc_layers, dec_layers, hparams)
                self.logits = tf.layers.dense(self.dec, num_pred_hours)


def transformer(inputs, targets, encoder_layers, decoder_layers, hparams):
    # Prepare encoder
    inputs = common_layers.flattend4d3d(inputs)
    output = transformer_prepare_encoder(
        inputs,
        hparams,
        features=None)
    enco_input, enco_self_att_bias, enco_deco_att_bias = output
    enco_input = tf.nn.dropout(
        enco_input, 1.0 - hparams.layer_prepost_dropout)
    # Prepare decoder
    decoder_input, decoder_self_attention_bias = transformer_prepare_decoder(
        targets,
        hparams)

    x = enco_input
    encoder_outputs = []
    with tf.variable_scope("encoder"):
        for layer_num, block_types in enumerate(encoder_layers):
            # Each encoder layers is composed of two blocks:
            # * self-attention block
            # * feed-forward block
            att_type, ff_type = block_types
            with tf.variable_scope("layer_{}".format(layer_num)):
                x = prepostprocess(layers[att_type])(
                    x,
                    bias=enco_self_att_bias,
                    name="att_{}".format(att_type))
                x = prepostprocess(layers[ff_type])(
                    x,
                    name="ff_{}".format(ff_type))
            encoder_outputs.append(x)

    x = decoder_input
    with tf.variable_scope("decoder"):
        for layer_num, block_types in enumerate(decoder_layers):
            # Each decoder layers is composed of three blocks:
            # * self-attention block
            # * enco-deco attention block (optional)
            # * feed-forward block
            self_att_type, att_ende_type, ff_type = block_types
            with tf.variable_scope("layer_{}".format(layer_num)):
                x = prepostprocess(layers[self_att_type])(
                    x,
                    bias=decoder_self_attention_bias,
                    name="self_att_{}".format(self_att_type))
                # Only add the enco-deco attention layer if there is an encoder
                if encoder_outputs:
                    x = prepostprocess(layers[att_ende_type])(
                        x,
                        memory_antecedent=encoder_outputs[-1],
                        bias=enco_deco_att_bias,
                        name="att_ende_{}".format(att_ende_type))
                x = prepostprocess(layers[ff_type])(
                    x,
                    name="ff_{}".format(ff_type))
        # If normalization is done in layer_preprocess, then it should also be
        # done on the output, since the output can grow very large, being the sum
        # of a whole stack of unnormalized layer outputs.
        x = preprocess(x, hparams)

    decoder_output = tf.expand_dims(x, axis=2)
    return decoder_output, encoder_outputs, cache["extra_loss"]
