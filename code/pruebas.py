import tensorflow as tf

from code.util import transformer_prepare_encoder, transformer_prepare_decoder
from tensor2tensor.common_attention import get_standarized_layers
from tensor2tensor import common_attention, common_layers
cache = dict(extra_loss=0.0)

# no norm no drop no nothing
def prepostprocess(fct):
    """Apply processing and capture the extra loss."""
    @add_var_scope()
    def decorated(x, *args, **kwargs):
        y, loss = fct(x, *args, **kwargs)
        cache["extra_loss"] += loss
        return x, y

layers = get_standarized_layers()
hparams = dict()

def encoder1(x, layerss, hparams):
    inputs = common_layers.flattend4d3d(x)
    output = transformer_prepare_encoder(
        inputs,
        hparams,
        features=None)
    enco_input, enco_self_att_bias, enco_deco_att_bias = output

    enco_input = tf.nn.dropout(
        enco_input, 1.0 - hparams.layer_prepost_dropout)
    
    x = enco_input
    encoder_outputs = []
    with tf.variable_scope("encoder"):
        for layer_num, block_types in enumerate(layerss):
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
        return encoder_outputs

def decoder1(x, layerss, hparams):
    targets = common_layers.flattend4d3d(x)
    output = transformer_prepare_decoder(
        targets, hparams, features=None)
    deco_input, deco_self_attention_bias = output
    deco_input = tf.nn.dropout(
        deco_input, 1.0 - hparams.layer_prepost_dropout)

    x = deco_input 
    with tf.variable_scope("decoder"):
        for layer_num, block_types in enumerate(layerss):
            # Each decoder layers is composed of three blocks:
            # * self-attention block
            # * enco-deco att block (optional)
            # * feed-forward blcok
            self_att_type, att_ende_type, ff_type = block_types
            with tf.variable_scope("layer_{}".format(layer_num)):
                x = prepostprocess(layers[self_att_type])(
                    x,
                    bias=deco_self_attention_bias,
                    name="self_att_{}".format(self_att_type))
                if encoder_outputs:
                    x = prepostprocess(layers[att_ende_type])(
                        x,
                        memory_antecedent=encoder_outputs[-1],
                        bias=encoder_decoder_attention_bias,
                        name="att_ende_{}".format(att_ende_type))
                x = prepostprocess(layers[ff_type])(
                    x,
                    name="ff_{}".format(ff_type))
    decoder_output = tf.expand_dims(x, 2)
    return decoder_output, cache["extra_loss"]
