
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import GRUCell, LSTMCell


from .components.positional import add_timing_signal_nd


class Encoder(object):
    """Class with a __call__ method that applies convolutions to an image"""

    def __init__(self, config):
        self._config = config


    def __call__(self, training, img, dropout):
        """Applies convolutions to the image
        Args:
            training: (tf.placeholder) tf.bool
            img: batch of img, shape = (?, height, width, channels), of type tf.uint8
        Returns:
            the encoded images, shape = (?, h', w', c')
        """
        img = tf.cast(img, tf.float32) - 128.
        img = img / 128.

        with tf.variable_scope("convolutional_encoder"):
            # conv + max pool -> /2
            out = tf.layers.conv2d(img, 64, 3, 1, "SAME", activation=tf.nn.relu)
            image_summary("out_1_layer", out)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            # conv + max pool -> /2
            out = tf.layers.conv2d(out, 128, 3, 1, "SAME", activation=tf.nn.relu)
            image_summary("out_2_layer", out)
            out = tf.layers.max_pooling2d(out, 2, 2, "SAME")

            # regular conv -> id
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)
            image_summary("out_3_layer", out)
            out = tf.layers.conv2d(out, 256, 3, 1, "SAME", activation=tf.nn.relu)
            image_summary("out_4_layer", out)
            if self._config.encoder_cnn == "vanilla":
                out = tf.layers.max_pooling2d(out, (2, 1), (2, 1), "SAME")

            out = tf.layers.conv2d(out, 512, 3, 1, "SAME", activation=tf.nn.relu)
            image_summary("out_5_layer", out)
            if self._config.encoder_cnn == "vanilla":
                out = tf.layers.max_pooling2d(out, (1, 2), (1, 2), "SAME")

            if self._config.encoder_cnn == "cnn":
                # conv with stride /2 (replaces the 2 max pool)
                out = tf.layers.conv2d(out, 512, (2, 4), 2, "SAME")

            # conv
            out = tf.layers.conv2d(out, 512, 3, 1, "VALID", activation=tf.nn.relu)
            image_summary("out_6_layer", out)
            if self._config.positional_embeddings:
                # from tensor2tensor lib - positional embeddings
                out = add_timing_signal_nd(out)
                image_summary("out_7_layer", out)
        return out

def image_summary(name_scope, tensor):
    with tf.variable_scope(name_scope):
        tf.summary.image("{}_{}".format(name_scope,0), tf.expand_dims(tensor[0,:,:,0], -1))
        # 磁盘炸了，只可视化一个
        # filter_count = tensor.shape[3]
        # for i in range(filter_count):
        #     tf.summary.image("{}_{}".format(name_scope,i), tf.expand_dims(tensor[:,:,:,i], -1))
            # tf.expand_dims(tensor[:,:,:,i], -1)
            # Tensor must be 4-D with last dim 1, 3, or 4, not [50,320], so we need to use expand_dims