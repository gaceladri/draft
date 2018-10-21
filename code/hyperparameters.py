import tensorflow as tf

FLAGS = tf.flags.FLAGS

# Model parameters
tf.flags.DEFINE_integer("hidden_size", 64, "Size of LSTM hidden layer.")
tf.flags.DEFINE_integer("memory_size", 16, "The number of memory slots.")
tf.flags.DEFINE_integer("word_size", 16, "The width of each memory slot.")
tf.flags.DEFINE_integer("num_write_heads", 1, "Number of memory write heads.")
tf.flags.DEFINE_integer("num_read_heads", 4, "Number of memory read heads.")
tf.flags.DEFINE_integer("clip_value", 20,
                        "Maximum absolute value of controller and dnc outputs.")

# Optimizer parameters.
tf.flags.DEFINE_float("max_grad_norm", 50, "Gradient clipping norm limit.")
tf.flags.DEFINE_float("learning_rate", 1e-4, "Optimizer learning rate.")
tf.flags.DEFINE_float("optimizer_epsilon", 1e-10,
                      "Epsilon used for RMSProp optimizer.")

# Task parameters
tf.flags.DEFINE_integer("batch_size", 16, "Batch size for training.")
tf.flags.DEFINE_integer("num_bits", 4, "Dimensionality of each vector to copy")
tf.flags.DEFINE_integer(
    "min_length", 1,
    "Lower limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer(
    "max_length", 2,
    "Upper limit on number of vectors in the observation pattern to copy")
tf.flags.DEFINE_integer("min_repeats", 1,
                        "Lower limit on number of copy repeats.")
tf.flags.DEFINE_integer("max_repeats", 2,
                        "Upper limit on number of copy repeats.")

