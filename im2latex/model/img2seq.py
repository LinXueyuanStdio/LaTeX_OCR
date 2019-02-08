import sys

import numpy as np

import tensorflow as tf
import tensorflow.contrib.layers as layers

from .base import BaseModel
from .decoder import Decoder
from .encoder import Encoder
from .evaluation.text import score_files, truncate_end, write_answers
from .utils.general import Config, Progbar, minibatches
from .utils.image import pad_batch_images
from .utils.text import pad_batch_formulas


class Img2SeqModel(BaseModel):
    """Specialized class for Img2Seq Model"""

    def __init__(self, config, dir_output, vocab):
        """
        Args:
            config: Config instance defining hyperparams
            vocab: Vocab instance defining useful vocab objects like tok_to_id

        """
        super(Img2SeqModel, self).__init__(config, dir_output)
        self._vocab = vocab


    def build_train(self, config):
        """Builds model"""
        self.logger.info("Building model...")

        self.encoder = Encoder(self._config)
        self.decoder = Decoder(self._config, self._vocab.n_tok, self._vocab.id_end)

        self._add_placeholders_op()
        self._add_pred_op()
        self._add_loss_op()

        self._add_train_op(config.lr_method, self.lr, self.loss, config.clip)
        self.init_session()

        self.logger.info("- done.")


    def build_pred(self):
        self.logger.info("Building model...")

        self.encoder = Encoder(self._config)
        self.decoder = Decoder(self._config, self._vocab.n_tok, self._vocab.id_end)

        self._add_placeholders_op()
        self._add_pred_op()
        self._add_loss_op()

        self.init_session()

        self.logger.info("- done.")



    def _add_placeholders_op(self):
        """
        Add placeholder attributes
        """
        # hyper params
        self.lr = tf.placeholder(tf.float32, shape=(), name='lr')
        self.dropout = tf.placeholder(tf.float32, shape=(), name='dropout')
        self.training = tf.placeholder(tf.bool, shape=(), name="training")


        # input of the graph
        self.img = tf.placeholder(tf.uint8, shape=(None, None, None, 1), name='img')
        self.formula = tf.placeholder(tf.int32, shape=(None, None), name='formula')
        self.formula_length = tf.placeholder(tf.int32, shape=(None, ), name='formula_length')

        # tensorboard
        tf.summary.scalar("lr", self.lr)


    def _get_feed_dict(self, img, training, formula=None, lr=None, dropout=1):
        """Returns a dict"""
        img = pad_batch_images(img)

        fd = {
            self.img: img,
            self.dropout: dropout,
            self.training: training,
        }

        if formula is not None:
            formula, formula_length = pad_batch_formulas(formula,
                    self._vocab.id_pad, self._vocab.id_end)
            # print img.shape, formula.shape
            fd[self.formula] = formula
            fd[self.formula_length] = formula_length
        if lr is not None:
            fd[self.lr] = lr

        return fd


    def _add_pred_op(self):
        """Defines self.pred"""
        encoded_img = self.encoder(self.training, self.img, self.dropout)
        train, test = self.decoder(self.training, encoded_img, self.formula,
                self.dropout)

        self.pred_train = train
        self.pred_test  = test


    def _add_loss_op(self):
        """Defines self.loss"""
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=self.pred_train, labels=self.formula)

        mask = tf.sequence_mask(self.formula_length)
        losses = tf.boolean_mask(losses, mask)

        # loss for training
        self.loss = tf.reduce_mean(losses)

        # # to compute perplexity for test
        self.ce_words = tf.reduce_sum(losses) # sum of CE for each word
        self.n_words = tf.reduce_sum(self.formula_length) # number of words

        # for tensorboard
        tf.summary.scalar("loss", self.loss)



    def _run_epoch(self, config, train_set, val_set, epoch, lr_schedule):
        """Performs an epoch of training

        Args:
            config: Config instance
            train_set: Dataset instance
            val_set: Dataset instance
            epoch: (int) id of the epoch, starting at 0
            lr_schedule: LRSchedule instance that takes care of learning proc

        Returns:
            score: (float) model will select weights that achieve the highest
                score

        """
        # logging
        batch_size = config.batch_size
        nbatches = (len(train_set) + batch_size - 1) // batch_size
        prog = Progbar(nbatches)

        # iterate over dataset
        for i, (img, formula) in enumerate(minibatches(train_set, batch_size)):
            # get feed dict
            fd = self._get_feed_dict(img, training=True, formula=formula,
                    lr=lr_schedule.lr, dropout=config.dropout)

            # update step
            _, loss_eval = self.sess.run([self.train_op, self.loss],
                    feed_dict=fd)
            prog.update(i + 1, [("loss", loss_eval), ("perplexity",
                    np.exp(loss_eval)), ("lr", lr_schedule.lr)])

            # update learning rate
            lr_schedule.update(batch_no=epoch*nbatches + i)

        # logging
        self.logger.info("- Training: {}".format(prog.info))

        # evaluation
        config_eval = Config({"dir_answers": self._dir_output + "formulas_val/",
                "batch_size": config.batch_size})
        scores = self.evaluate(config_eval, val_set)
        score = scores[config.metric_val]
        lr_schedule.update(score=score)

        return score



    def write_prediction(self, config, test_set):
        """Performs an epoch of evaluation

        Args:
            config: (Config) with batch_size and dir_answers
            test_set:(Dataset) instance

        Returns:
            files: (list) of path to files
            perp: (float) perplexity on test set

        """
        # initialize containers of references and predictions
        if self._config.decoding == "greedy":
            refs, hyps = [], [[]]
        elif self._config.decoding == "beam_search":
            refs, hyps = [], [[] for i in range(self._config.beam_size)]

        # iterate over the dataset
        n_words, ce_words = 0, 0 # sum of ce for all words + nb of words
        for img, formula in minibatches(test_set, config.batch_size):
            fd = self._get_feed_dict(img, training=False, formula=formula,
                    dropout=1)
            ce_words_eval, n_words_eval, ids_eval = self.sess.run(
                    [self.ce_words, self.n_words, self.pred_test.ids],
                    feed_dict=fd)

            # TODO(guillaume): move this logic into tf graph
            if self._config.decoding == "greedy":
                ids_eval = np.expand_dims(ids_eval, axis=1)

            elif self._config.decoding == "beam_search":
                ids_eval = np.transpose(ids_eval, [0, 2, 1])

            n_words += n_words_eval
            ce_words += ce_words_eval
            for form, preds in zip(formula, ids_eval):
                refs.append(form)
                for i, pred in enumerate(preds):
                    hyps[i].append(pred)

        files = write_answers(refs, hyps, self._vocab.id_to_tok,
                config.dir_answers, self._vocab.id_end)

        perp = - np.exp(ce_words / float(n_words))

        return files, perp


    def _run_evaluate(self, config, test_set):
        """Performs an epoch of evaluation

        Args:
            test_set: Dataset instance
            params: (dict) with extra params in it
                - "dir_name": (string)

        Returns:
            scores: (dict) scores["acc"] = 0.85 for instance

        """
        files, perp = self.write_prediction(config, test_set)
        scores = score_files(files[0], files[1])
        scores["perplexity"] = perp

        return scores


    def predict_batch(self, images):
        if self._config.decoding == "greedy":
            hyps = [[]]
        elif self._config.decoding == "beam_search":
            hyps = [[] for i in range(self._config.beam_size)]

        fd = self._get_feed_dict(images, training=False, dropout=1)
        ids_eval, = self.sess.run([self.pred_test.ids], feed_dict=fd)

        if self._config.decoding == "greedy":
            ids_eval = np.expand_dims(ids_eval, axis=1)

        elif self._config.decoding == "beam_search":
            ids_eval = np.transpose(ids_eval, [0, 2, 1])

        for preds in ids_eval:
            for i, pred in enumerate(preds):
                p = truncate_end(pred, self._vocab.id_end)
                p = " ".join([self._vocab.id_to_tok[idx] for idx in p])
                hyps[i].append(p)

        return hyps


    def predict(self, img):
        preds = self.predict_batch([img])
        preds_ = []
        # extract only one element (no batch)
        for hyp in preds:
            preds_.append(hyp[0])

        return preds_
