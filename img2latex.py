from __future__ import division
import math
import time
import logging
import collections
import numpy as np
import tensorflow as tf
from six import xrange
import sys
import os

from tensorflow.contrib.rnn import GRUCell, LSTMCell
from tensorflow.contrib.rnn import RNNCell, LSTMStateTuple


config = {
    "encoder_cnn": "vanilla",
    "positional_embeddings": True,

    "attn_cell_config": {
        "cell_type": "lstm",
        "num_units": 512,
        "dim_e": 512,
        "dim_o": 512,
        "dim_embeddings": 80
    },

    "decoding": "beam_search",
    "beam_size": 5,
    "div_gamma": 1,
    "div_prob": 0,

    "max_length_formula": 150
}


def get_logger(filename):
    """Return instance of logger"""
    logger = logging.getLogger('logger')
    logger.setLevel(logging.INFO)
    logging.basicConfig(format='%(message)s', level=logging.INFO)
    handler = logging.FileHandler(filename)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(
        '%(asctime)s:%(levelname)s: %(message)s'))
    logging.getLogger().addHandler(handler)
    return logger


def add_timing_signal_nd(x, min_timescale=1.0, max_timescale=1.0e4):
    """为了解决img扁平化为向量后，公式符号之间的位置关系丢失，需要引入Attention来重新给扁平化的向量加上去
    Adds a bunch of sinusoids of different frequencies to a Tensor.

    Each channel of the input Tensor is incremented by a sinusoid of a difft
    frequency and phase in one of the positional dimensions.

    This allows attention to learn to use absolute and relative positions.
    Timing signals should be added to some precursors of both the query and the
    memory inputs to attention.

    The use of relative position is possible because sin(a+b) and cos(a+b) can
    be experessed in terms of b, sin(a) and cos(a).

    x is a Tensor with n "positional" dimensions, e.g. one dimension for a
    sequence or two dimensions for an image

    We use a geometric sequence of timescales starting with
    min_timescale and ending with max_timescale.  The number of different
    timescales is equal to channels // (n * 2). For each timescale, we
    generate the two sinusoidal signals sin(timestep/timescale) and
    cos(timestep/timescale).  All of these sinusoids are concatenated in
    the channels dimension.

    Args:
        x: a Tensor with shape [batch, d1 ... dn, channels]
        min_timescale: a float
        max_timescale: a float

    Returns:
        a Tensor the same shape as x.

    """
    static_shape = x.get_shape().as_list()
    num_dims = len(static_shape) - 2
    channels = tf.shape(x)[-1]
    num_timescales = channels // (num_dims * 2)
    log_timescale_increment = (
        math.log(float(max_timescale) / float(min_timescale)) /
        (tf.to_float(num_timescales) - 1))
    inv_timescales = min_timescale * tf.exp(
        tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)
    for dim in xrange(num_dims):
        length = tf.shape(x)[dim + 1]
        position = tf.to_float(tf.range(length))
        scaled_time = tf.expand_dims(position, 1) * tf.expand_dims(
            inv_timescales, 0)
        signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=1)
        prepad = dim * 2 * num_timescales
        postpad = channels - (dim + 1) * 2 * num_timescales
        signal = tf.pad(signal, [[0, 0], [prepad, postpad]])
        for _ in xrange(1 + dim):
            signal = tf.expand_dims(signal, 0)
        for _ in xrange(num_dims - 1 - dim):
            signal = tf.expand_dims(signal, -2)
        x += signal
    return x


def get_embeddings(formula, E, dim, start_token, batch_size):
    """Returns the embedding of the n-1 first elements in the formula concat
    with the start token

    Args:
        formula: (tf.placeholder) tf.uint32
        E: tf.Variable (matrix)
        dim: (int) dimension of embeddings
        start_token: tf.Variable
        batch_size: tf variable extracted from placeholder

    Returns:
        embeddings_train: tensor

    """
    formula_ = tf.nn.embedding_lookup(E, formula)
    start_token_ = tf.reshape(start_token, [1, 1, dim])
    start_tokens = tf.tile(start_token_, multiples=[batch_size, 1, 1])
    embeddings = tf.concat([start_tokens, formula_[:, :-1, :]], axis=1)

    return embeddings


def embedding_initializer():
    """Returns initializer for embeddings"""
    def _initializer(shape, dtype, partition_info=None):
        E = tf.random_uniform(shape, minval=-1.0, maxval=1.0, dtype=dtype)
        E = tf.nn.l2_normalize(E, -1)
        return E

    return _initializer


AttentionState = collections.namedtuple("AttentionState", ("cell_state", "o"))


class AttentionCell(RNNCell):
    def __init__(self, cell, attention_mechanism, dropout, attn_cell_config,
                 num_proj, dtype=tf.float32):
        """
        Args:
            cell: (RNNCell)
            attention_mechanism: (AttentionMechanism)
            dropout: (tf.float)
            attn_cell_config: (dict) hyper params

        """
        # variables and tensors
        self._cell = cell
        self._attention_mechanism = attention_mechanism
        self._dropout = dropout

        # hyperparameters and shapes
        self._n_channels = self._attention_mechanism._n_channels
        self._dim_e = attn_cell_config["dim_e"]
        self._dim_o = attn_cell_config["dim_o"]
        self._num_units = attn_cell_config["num_units"]
        self._dim_embeddings = attn_cell_config["dim_embeddings"]
        self._num_proj = num_proj
        self._dtype = dtype

        # for RNNCell
        self._state_size = AttentionState(self._cell._state_size, self._dim_o)

    @property
    def state_size(self):
        return self._state_size

    @property
    def output_size(self):
        return self._num_proj

    @property
    def output_dtype(self):
        return self._dtype

    def initial_state(self):
        """Returns initial state for the lstm"""
        initial_cell_state = self._attention_mechanism.initial_cell_state(self._cell)
        initial_o = self._attention_mechanism.initial_state("o", self._dim_o)

        return AttentionState(initial_cell_state, initial_o)

    def step(self, embedding, attn_cell_state):
        """
        Args:
            embedding: shape = (batch_size, dim_embeddings) embeddings
                from previous time step
            attn_cell_state: (AttentionState) state from previous time step

        """
        prev_cell_state, o = attn_cell_state

        scope = tf.get_variable_scope()
        with tf.variable_scope(scope):
            # compute new h
            x = tf.concat([embedding, o], axis=-1)
            new_h, new_cell_state = self._cell.__call__(x, prev_cell_state)
            new_h = tf.nn.dropout(new_h, self._dropout)

            # compute attention
            c = self._attention_mechanism.context(new_h)

            # compute o
            o_W_c = tf.get_variable("o_W_c", dtype=tf.float32,
                                    shape=(self._n_channels, self._dim_o))
            o_W_h = tf.get_variable("o_W_h", dtype=tf.float32,
                                    shape=(self._num_units, self._dim_o))

            new_o = tf.tanh(tf.matmul(new_h, o_W_h) + tf.matmul(c, o_W_c))
            new_o = tf.nn.dropout(new_o, self._dropout)

            y_W_o = tf.get_variable("y_W_o", dtype=tf.float32,
                                    shape=(self._dim_o, self._num_proj))
            logits = tf.matmul(new_o, y_W_o)

            # new Attn cell state
            new_state = AttentionState(new_cell_state, new_o)

            return logits, new_state

    def __call__(self, inputs, state):
        """
        Args:
            inputs: the embedding of the previous word for training only
            state: (AttentionState) (h, o) where h is the hidden state and
                o is the vector used to make the prediction of
                the previous word

        """
        new_output, new_state = self.step(inputs, state)

        return (new_output, new_state)


class AttentionMechanism(object):
    """Class to compute attention over an image"""

    def __init__(self, img, dim_e, tiles=1):
        """Stores the image under the right shape.

        We loose the H, W dimensions and merge them into a single
        dimension that corresponds to "regions" of the image.

        Args:
            img: (tf.Tensor) image
            dim_e: (int) dimension of the intermediary vector used to
                compute attention
            tiles: (int) default 1, input to context h may have size
                    (tile * batch_size, ...)

        """
        if len(img.shape) == 3:
            self._img = img
        elif len(img.shape) == 4:
            N = tf.shape(img)[0]
            H, W = tf.shape(img)[1], tf.shape(img)[2]  # image
            C = img.shape[3].value                 # channels
            self._img = tf.reshape(img, shape=[N, H*W, C])
        else:
            print("Image shape not supported")
            raise NotImplementedError

        # dimensions
        self._n_regions = tf.shape(self._img)[1]
        self._n_channels = self._img.shape[2].value
        self._dim_e = dim_e
        self._tiles = tiles
        self._scope_name = "att_mechanism"

        # attention vector over the image
        self._att_img = tf.layers.dense(
            inputs=self._img,
            units=self._dim_e,
            use_bias=False,
            name="att_img")

    def context(self, h):
        """Computes attention

        Args:
            h: (batch_size, num_units) hidden state

        Returns:
            c: (batch_size, channels) context vector

        """
        with tf.variable_scope(self._scope_name):
            if self._tiles > 1:
                att_img = tf.expand_dims(self._att_img, axis=1)
                att_img = tf.tile(att_img, multiples=[1, self._tiles, 1, 1])
                att_img = tf.reshape(att_img, shape=[-1, self._n_regions,
                                                     self._dim_e])
                img = tf.expand_dims(self._img, axis=1)
                img = tf.tile(img, multiples=[1, self._tiles, 1, 1])
                img = tf.reshape(img, shape=[-1, self._n_regions,
                                             self._n_channels])
            else:
                att_img = self._att_img
                img = self._img

            # computes attention over the hidden vector
            att_h = tf.layers.dense(inputs=h, units=self._dim_e, use_bias=False)

            # sums the two contributions
            att_h = tf.expand_dims(att_h, axis=1)
            att = tf.tanh(att_img + att_h)

            # computes scalar product with beta vector
            # works faster with a matmul than with a * and a tf.reduce_sum
            att_beta = tf.get_variable("att_beta", shape=[self._dim_e, 1],
                                       dtype=tf.float32)
            att_flat = tf.reshape(att, shape=[-1, self._dim_e])
            e = tf.matmul(att_flat, att_beta)
            e = tf.reshape(e, shape=[-1, self._n_regions])

            # compute weights
            a = tf.nn.softmax(e)
            a = tf.expand_dims(a, axis=-1)
            c = tf.reduce_sum(a * img, axis=1)

            return c

    def initial_cell_state(self, cell):
        """Returns initial state of a cell computed from the image

        Assumes cell.state_type is an instance of named_tuple.
        Ex: LSTMStateTuple

        Args:
            cell: (instance of RNNCell) must define _state_size

        """
        _states_0 = []
        for hidden_name in cell._state_size._fields:
            hidden_dim = getattr(cell._state_size, hidden_name)
            h = self.initial_state(hidden_name, hidden_dim)
            _states_0.append(h)

        initial_state_cell = type(cell.state_size)(*_states_0)

        return initial_state_cell

    def initial_state(self, name, dim):
        """Returns initial state of dimension specified by dim"""
        with tf.variable_scope(self._scope_name):
            img_mean = tf.reduce_mean(self._img, axis=1)
            W = tf.get_variable("W_{}_0".format(name), shape=[self._n_channels,
                                                              dim])
            b = tf.get_variable("b_{}_0".format(name), shape=[dim])
            h = tf.tanh(tf.matmul(img_mean, W) + b)

            return h


class LRSchedule(object):
    """Class for Learning Rate schedules

    Implements
        - (time) exponential decay with custom range
            - needs to set start_decay, end_decay, lr_init and lr_min
            - set end_decay to None to deactivate
        - (time) warm start:
            - needs to set lr_warm, end_warm.
            - set end_warm to None to deactivate
        - (score) mult decay if no improvement over score
            - needs to set decay_rate
            - set decay_rate to None to deactivate
        - (score) early stopping if no imprv
            - needs to set early_stopping
            - set early_stopping to None to deactivate

    All durations are measured in number of batches
    For usage, must call the update function at each batch.
    You can access the current learning rate with self.lr

    """

    def __init__(self, lr_init=1e-3, lr_min=1e-4, start_decay=0,
                 decay_rate=None, end_decay=None, lr_warm=1e-4, end_warm=None,
                 early_stopping=None):
        """Initializes Learning Rate schedule

        Sets self.lr and self.stop_training

        Args:
            lr_init: (float) initial lr
            lr_min: (float)
            start_decay: (int) id of batch to start decay
            decay_rate: (float) lr *= decay_rate if no improval. If None, no
                multiplicative decay.
            end_decay: (int) id of batch to end decay. If None, no exp decay
            lr_warm: (float) constant learning rate at the beginning
            end_warm: (int) id of batch to keep the lr_warm before returning to
                lr_init and start the regular schedule.
            early_stopping: (int) number of batches with no imprv

        """
        self._lr_init = lr_init
        self._lr_min = lr_min
        self._start_decay = start_decay
        self._decay_rate = decay_rate
        self._end_decay = end_decay
        self._lr_warm = lr_warm
        self._end_warm = end_warm

        self._score = None
        self._early_stopping = early_stopping
        self._n_batch_no_imprv = 0

        # warm start initializes learning rate to warm start
        if self._end_warm is not None:
            # make sure that decay happens after the warm up
            self._start_decay = max(self._end_warm, self._start_decay)
            self.lr = self._lr_warm
        else:
            self.lr = lr_init

        # setup of exponential decay
        if self._end_decay is not None:
            self._exp_decay = np.power(lr_min/lr_init,
                                       1/float(self._end_decay - self._start_decay))

    @property
    def stop_training(self):
        """For Early Stopping"""
        if (self._early_stopping is not None and
                (self._n_batch_no_imprv >= self._early_stopping)):
            return True
        else:
            return False

    def update(self, batch_no=None, score=None):
        """Updates the learning rate

        (score) decay by self.decay rate if score is higher than previous
        (time) update lr according to
            - warm up
            - exp decay
        Both updates can concurrently happen

        Args:
            batch_no: (int) id of the batch
            score: (float) score, higher is better

        """
        # update based on time
        if batch_no is not None:
            if (self._end_warm is not None and
                    (self._end_warm <= batch_no <= self._start_decay)):
                self.lr = self._lr_init

            if batch_no > self._start_decay and self._end_decay is not None:
                self.lr *= self._exp_decay

        # update based on performance
        if self._decay_rate is not None:
            if score is not None and self._score is not None:
                if score <= self._score:
                    self.lr *= self._decay_rate
                    self._n_batch_no_imprv += 1
                else:
                    self._n_batch_no_imprv = 0

        # update last score eval
        if score is not None:
            self._score = score

        self.lr = max(self.lr, self._lr_min)


class Progbar(object):
    """Progbar class inspired by keras"""

    def __init__(self, max_step, width=30):
        self.max_step = max_step
        self.width = width
        self.last_width = 0

        self.sum_values = {}

        self.start = time.time()
        self.last_step = 0

        self.info = ""
        self.bar = ""

    def _update_values(self, curr_step, values):
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (curr_step - self.last_step),
                                      curr_step - self.last_step]
            else:
                self.sum_values[k][0] += v * (curr_step - self.last_step)
                self.sum_values[k][1] += (curr_step - self.last_step)

    def _write_bar(self, curr_step):
        last_width = self.last_width
        sys.stdout.write("\b" * last_width)
        sys.stdout.write("\r")

        numdigits = int(np.floor(np.log10(self.max_step))) + 1
        barstr = '%%%dd/%%%dd [' % (numdigits, numdigits)
        bar = barstr % (curr_step, self.max_step)
        prog = float(curr_step)/self.max_step
        prog_width = int(self.width*prog)
        if prog_width > 0:
            bar += ('='*(prog_width-1))
            if curr_step < self.max_step:
                bar += '>'
            else:
                bar += '='
        bar += ('.'*(self.width-prog_width))
        bar += ']'
        sys.stdout.write(bar)

        return bar

    def _get_eta(self, curr_step):
        now = time.time()
        if curr_step:
            time_per_unit = (now - self.start) / curr_step
        else:
            time_per_unit = 0
        eta = time_per_unit*(self.max_step - curr_step)

        if curr_step < self.max_step:
            info = ' - ETA: %ds' % eta
        else:
            info = ' - %ds' % (now - self.start)

        return info

    def _get_values_sum(self):
        info = ""
        for name, value in self.sum_values.items():
            info += ' - %s: %.4f' % (name, value[0] / max(1, value[1]))
        return info

    def _write_info(self, curr_step):
        info = ""
        info += self._get_eta(curr_step)
        info += self._get_values_sum()

        sys.stdout.write(info)

        return info

    def _update_width(self, curr_step):
        curr_width = len(self.bar) + len(self.info)
        if curr_width < self.last_width:
            sys.stdout.write(" "*(self.last_width - curr_width))

        if curr_step >= self.max_step:
            sys.stdout.write("\n")

        sys.stdout.flush()

        self.last_width = curr_width

    def update(self, curr_step, values):
        """Updates the progress bar.

        Args:
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.

        """
        self._update_values(curr_step, values)
        self.bar = self._write_bar(curr_step)
        self.info = self._write_info(curr_step)
        self._update_width(curr_step)
        self.last_step = curr_step


# region 输入 placeholder
lr = tf.placeholder(tf.float32, shape=(),  name='lr')
dropout = tf.placeholder(tf.float32, shape=(),  name='dropout')
training = tf.placeholder(tf.bool, shape=(),  name="training")

# input of the graph
img = tf.placeholder(tf.uint8, shape=(None, None, None, 1),  name='img')
formula = tf.placeholder(tf.int32, shape=(None, None),   name='formula')
formula_length = tf.placeholder(tf.int32, shape=(None, ),  name='formula_length')

# tensorboard
tf.summary.scalar("lr", lr)
# endregion

# 卷积层
img = tf.cast(img, tf.float32) / 255.

with tf.variable_scope("convolutional_encoder"):
    # conv + max pool -> /2
    encoded_img = tf.layers.conv2d(img, 64, 3, 1, "SAME", activation=tf.nn.relu)
    encoded_img = tf.layers.max_pooling2d(encoded_img, 2, 2, "SAME")

    # conv + max pool -> /2
    encoded_img = tf.layers.conv2d(encoded_img, 128, 3, 1, "SAME", activation=tf.nn.relu)
    encoded_img = tf.layers.max_pooling2d(encoded_img, 2, 2, "SAME")

    # regular conv -> id
    encoded_img = tf.layers.conv2d(encoded_img, 256, 3, 1, "SAME", activation=tf.nn.relu)

    encoded_img = tf.layers.conv2d(encoded_img, 256, 3, 1, "SAME", activation=tf.nn.relu)

    if config.encoder_cnn == "vanilla":
        encoded_img = tf.layers.max_pooling2d(encoded_img, (2, 1), (2, 1), "SAME")

    encoded_img = tf.layers.conv2d(encoded_img, 512, 3, 1, "SAME", activation=tf.nn.relu)

    if config.encoder_cnn == "vanilla":
        encoded_img = tf.layers.max_pooling2d(encoded_img, (1, 2), (1, 2), "SAME")

    if config.encoder_cnn == "cnn":
        # conv with stride /2 (replaces the 2 max pool)
        encoded_img = tf.layers.conv2d(encoded_img, 512, (2, 4), 2, "SAME")

    # conv
    encoded_img = tf.layers.conv2d(encoded_img, 512, 3, 1, "VALID", activation=tf.nn.relu)

    if config.positional_embeddings:
        # from tensor2tensor lib - positional embeddings
        encoded_img = add_timing_signal_nd(encoded_img)
# 现在 encoded_img 是卷积后的了


dim_embeddings = config["attn_cell_config"]["dim_embeddings"]
E = tf.get_variable("E", initializer=embedding_initializer(), shape=[n_tok, dim_embeddings], dtype=tf.float32)

start_token = tf.get_variable("start_token", dtype=tf.float32, shape=[dim_embeddings], initializer=embedding_initializer())

batch_size = tf.shape(encoded_img)[0]

# training
with tf.variable_scope("attn_cell", reuse=False):
    embeddings = get_embeddings(formula, E, dim_embeddings, start_token, batch_size)
    attn_meca = AttentionMechanism(encoded_img, config.attn_cell_config["dim_e"])
    recu_cell = LSTMCell(config.attn_cell_config["num_units"])
    attn_cell = AttentionCell(recu_cell, attn_meca, dropout, config.attn_cell_config, n_tok)

    train_outputs, _ = tf.nn.dynamic_rnn(attn_cell, embeddings, initial_state=attn_cell.initial_state())

# decoding
with tf.variable_scope("attn_cell", reuse=True):
    attn_meca = AttentionMechanism(img=encoded_img,
                                   dim_e=config.attn_cell_config["dim_e"],
                                   tiles=tiles)
    recu_cell = LSTMCell(config.attn_cell_config["num_units"], reuse=True)
    attn_cell = AttentionCell(recu_cell, attn_meca, dropout, config.attn_cell_config, n_tok)
    if config.decoding == "greedy":
        decoder_cell = GreedyDecoderCell(E, attn_cell, batch_size, start_token, id_end)
    elif config.decoding == "beam_search":
        decoder_cell = BeamSearchDecoderCell(E, attn_cell, batch_size,   start_token, id_end, config.beam_size,
                                             config.div_gamma, config.div_prob)

    test_outputs, _ = dynamic_decode(decoder_cell, config.max_length_formula+1)

# train_outputs, test_outputs

# region 定义损失函数
losses = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=train_outputs, labels=formula)

mask = tf.sequence_mask(formula_length)
losses = tf.boolean_mask(losses, mask)

# loss for training
loss = tf.reduce_mean(losses)

# # to compute perplexity for test
ce_words = tf.reduce_sum(losses)  # sum of CE for each word
n_words = tf.reduce_sum(formula_length)  # number of words

# for tensorboard
tf.summary.scalar("loss", loss)
# endregion

# region 定义优化器
optimizer = tf.train.AdamOptimizer(lr)
# for batch norm beta gamma
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    clip = config.clip
    if clip > 0:  # gradient clipping if clip is positive
        grads, vs = zip(*optimizer.compute_gradients(loss))
        grads, gnorm = tf.clip_by_global_norm(grads, clip)
        train_op = optimizer.apply_gradients(zip(grads, vs))
    else:
        train_op = optimizer.minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver()

# endregion

best_score = None
logger = get_logger("/log/model.log")
n_batches_epoch = ((len(train_set) + config.batch_size - 1) //
                   config.batch_size)
lr_schedule = LRSchedule(lr_init=config.lr_init,
                         start_decay=config.start_decay*n_batches_epoch,
                         end_decay=config.end_decay*n_batches_epoch,
                         end_warm=config.end_warm*n_batches_epoch,
                         lr_warm=config.lr_warm,
                         lr_min=config.lr_min)
for epoch in range(config.n_epochs):
        # logging
    tic = time.time()
    logger.info("Epoch {:}/{:}".format(epoch+1, config.n_epochs))

    # epoch
    score = _run_epoch(config, train_set, val_set, epoch, lr_schedule)
    # logging
    batch_size = config.batch_size
    nbatches = (len(train_set) + batch_size - 1) // batch_size
    prog = Progbar(nbatches)

    # iterate over dataset
    for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
        # get feed dict
        fd = _get_feed_dict(img, training=True, formula=formula,
                            lr=lr_schedule.lr, dropout=config.dropout)

        # update step
        _, loss_eval = sess.run([train_op, loss],     feed_dict=fd)
        prog.update(i + 1, [("loss", loss_eval), ("perplexity",
                                                  np.exp(loss_eval)), ("lr", lr_schedule.lr)])

        # update learning rate
        lr_schedule.update(batch_no=epoch*nbatches + i)

        # logging
        logger.info("- Training: {}".format(prog.info))

        # evaluation
        config_eval = Config({"dir_answers": _dir_output + "formulas_val/",
                              "batch_size": config.batch_size})
        scores = evaluate(config_eval, val_set)
        score = scores[config.metric_val]
        lr_schedule.update(score=score)

        # return score

    # save weights if we have new best score on eval
    if best_score is None or score >= best_score:
        best_score = score
        logger.info("- New best score ({:04.2f})!".format(best_score))
        dir_model = "model/weights/"
        if dir_model is not None:
            if not os.path.exists(dir_model):
                os.makedirs(dir_model)

        # logging
        sys.stdout.write("\r- Saving model...")
        sys.stdout.flush()

        # saving
        saver.save(sess, dir_model)

        # logging
        sys.stdout.write("\r")
        sys.stdout.flush()
        logger.info("- Saved model in {}".format(dir_model))

    # logging
    toc = time.time()
    logger.info("- Elapsed time: {:04.2f}, lr: {:04.5f}".format(toc-tic, lr_schedule.lr))

# return best_score
