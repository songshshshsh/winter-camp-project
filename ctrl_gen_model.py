# Copyright 2018 The Texar Authors. All Rights Reserved.
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
"""Text style transfer
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals

import tensorflow as tf

import texar as tx
from texar.modules import WordEmbedder, UnidirectionalRNNEncoder, \
        MLPTransformConnector, AttentionRNNDecoder, \
        GumbelSoftmaxEmbeddingHelper, Conv1DClassifier
from texar.core import get_train_op
from texar.utils import collect_trainable_variables, get_batch_size
import sys


class CtrlGenModel(object):
    """Control
    """

    def __init__(self, inputs, vocab, gamma, lambda_g, hparams=None):
        self._hparams = tx.HParams(hparams, None)
        print(inputs)
        self._build_model(inputs, vocab, gamma, lambda_g)

    def _high_level_classifier(self, classifier, clas_embedder, inputs, vocab, gamma, lambda_g, classifier_id, classifier_soft_id, classifier_seq_len):
        # Classification loss for the classifier
        print('classifier', classifier_id, classifier_soft_id)
        classifier0, classifier1, classifier2, classifier3 = classifier
        clas_logits0, clas_preds0 = classifier0(
            inputs=clas_embedder(ids=classifier_id, soft_ids=classifier_soft_id),
            sequence_length=classifier_seq_len)
        clas_logits1, clas_preds1 = classifier1(
            inputs=clas_embedder(ids=classifier_id, soft_ids=classifier_soft_id),
            sequence_length=classifier_seq_len)
        clas_logits2, clas_preds2 = classifier2(
            inputs=clas_embedder(ids=classifier_id, soft_ids=classifier_soft_id),
            sequence_length=classifier_seq_len)
        clas_logits3, clas_preds3 = classifier3(
            inputs=clas_embedder(ids=classifier_id, soft_ids=classifier_soft_id),
            sequence_length=classifier_seq_len)
        clas_logits0 = tf.reshape(clas_logits0, (-1, 1))
        clas_logits1 = tf.reshape(clas_logits1, (-1, 1))
        clas_logits2 = tf.reshape(clas_logits2, (-1, 1))
        clas_logits3 = tf.reshape(clas_logits3, (-1, 1))
        clas_preds0 = tf.reshape(clas_preds0, (-1, 1))
        clas_preds1 = tf.reshape(clas_preds1, (-1, 1))
        clas_preds2 = tf.reshape(clas_preds2, (-1, 1))
        clas_preds3 = tf.reshape(clas_preds3, (-1, 1))
        print('clas_logits', clas_logits0, 'clas_preds', clas_preds0)
        clas_logits = tf.concat([clas_logits0, clas_logits1, clas_logits2, clas_logits3], axis = 1)
        clas_preds = tf.concat([clas_preds0, clas_preds1, clas_preds2, clas_preds3], axis = 1)
        print('clas', clas_logits, clas_preds)
        sys.stdout.flush()
        return clas_logits, clas_preds

    def _build_model(self, inputs, vocab, gamma, lambda_g):
        """Builds the model.
        """
        embedder = WordEmbedder(
            vocab_size=vocab.size,
            hparams=self._hparams.embedder)
        encoder = UnidirectionalRNNEncoder(hparams=self._hparams.encoder)

        # text_ids for encoder, with BOS token removed
        enc_text_ids = inputs['text_ids'][:, 1:]
        enc_outputs, final_state = encoder(embedder(enc_text_ids),
                                           sequence_length=inputs['length']-1)
        z = final_state[:, self._hparams.dim_c:]

        # Encodes label
        label_connector = MLPTransformConnector(self._hparams.dim_c)

        # Gets the sentence representation: h = (c, z)
        labels0 = tf.to_float(tf.reshape(inputs['labels0'], [-1, 1]))
        labels1 = tf.to_float(tf.reshape(inputs['labels1'], [-1, 1]))
        labels2 = tf.to_float(tf.reshape(inputs['labels2'], [-1, 1]))
        labels3 = tf.to_float(tf.reshape(inputs['labels3'], [-1, 1]))
        labels = tf.concat([labels0, labels1, labels2, labels3], axis = 1)
        print('labels', labels)
        sys.stdout.flush()
        c = label_connector(labels)
        c_ = label_connector(1 - labels)
        h = tf.concat([c, z], 1)
        h_ = tf.concat([c_, z], 1)

        # Teacher-force decoding and the auto-encoding loss for G
        decoder = AttentionRNNDecoder(
            memory=enc_outputs,
            memory_sequence_length=inputs['length']-1,
            cell_input_fn=lambda inputs, attention: inputs,
            vocab_size=vocab.size,
            hparams=self._hparams.decoder)

        connector = MLPTransformConnector(decoder.state_size)

        g_outputs, _, _ = decoder(
            initial_state=connector(h), inputs=inputs['text_ids'],
            embedding=embedder, sequence_length=inputs['length']-1)

        print('labels shape', inputs['text_ids'][:, 1:], 'logits shape', g_outputs.logits)
        print(inputs['length'] - 1)
        loss_g_ae = tx.losses.sequence_sparse_softmax_cross_entropy(
            labels=inputs['text_ids'][:, 1:],
            logits=g_outputs.logits,
            sequence_length=inputs['length']-1,
            average_across_timesteps=True,
            sum_over_timesteps=False)

        # Gumbel-softmax decoding, used in training
        start_tokens = tf.ones_like(inputs['labels0']) * vocab.bos_token_id
        end_token = vocab.eos_token_id
        gumbel_helper = GumbelSoftmaxEmbeddingHelper(
            embedder.embedding, start_tokens, end_token, gamma)

        soft_outputs_, _, soft_length_, = decoder(
            helper=gumbel_helper, initial_state=connector(h_))

        print(g_outputs, soft_outputs_)

        # Greedy decoding, used in eval
        outputs_, _, length_ = decoder(
            decoding_strategy='infer_greedy', initial_state=connector(h_),
            embedding=embedder, start_tokens=start_tokens, end_token=end_token)
        # Creates classifier
        classifier0 = Conv1DClassifier(hparams=self._hparams.classifier)
        classifier1 = Conv1DClassifier(hparams=self._hparams.classifier)
        classifier2 = Conv1DClassifier(hparams=self._hparams.classifier)
        classifier3 = Conv1DClassifier(hparams=self._hparams.classifier)
        clas_embedder = WordEmbedder(vocab_size=vocab.size,
                                     hparams=self._hparams.embedder)

        clas_logits, clas_preds = self._high_level_classifier([classifier0, classifier1, classifier2, classifier3],
            clas_embedder, inputs, vocab, gamma, lambda_g, inputs['text_ids'][:, 1:], None, inputs['length']-1)
        loss_d_clas = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(labels), logits=clas_logits)
        loss_d_clas = tf.reduce_mean(loss_d_clas)
        accu_d = tx.evals.accuracy(labels, preds=clas_preds)

        # Classification loss for the generator, based on soft samples
        # soft_logits, soft_preds = classifier(
        #     inputs=clas_embedder(soft_ids=soft_outputs_.sample_id),
        #     sequence_length=soft_length_)
        soft_logits, soft_preds = self._high_level_classifier([classifier0, classifier1, classifier2, classifier3],
            clas_embedder, inputs, vocab, gamma, lambda_g, None, soft_outputs_.sample_id, soft_length_)
        print(soft_logits.shape, soft_preds.shape)
        loss_g_clas = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=tf.to_float(1-labels), logits=soft_logits)
        loss_g_clas = tf.reduce_mean(loss_g_clas)

        # Accuracy on soft samples, for training progress monitoring
        accu_g = tx.evals.accuracy(labels=1-labels, preds=soft_preds)

        # Accuracy on greedy-decoded samples, for training progress monitoring
        # _, gdy_preds = classifier(
        #     inputs=clas_embedder(ids=outputs_.sample_id),
        #     sequence_length=length_)
        _, gdy_preds = self._high_level_classifier([classifier0, classifier1, classifier2, classifier3],
            clas_embedder, inputs, vocab, gamma, lambda_g, outputs_.sample_id, None, length_)
        print(gdy_preds.shape)
        accu_g_gdy = tx.evals.accuracy(
            labels=1-labels, preds=gdy_preds)

        # Aggregates losses
        loss_g = loss_g_ae + lambda_g * loss_g_clas
        loss_d = loss_d_clas

        # Creates optimizers
        g_vars = collect_trainable_variables(
            [embedder, encoder, label_connector, connector, decoder])
        d_vars = collect_trainable_variables([clas_embedder, classifier0, classifier1, classifier2, classifier3])

        train_op_g = get_train_op(
            loss_g, g_vars, hparams=self._hparams.opt)
        train_op_g_ae = get_train_op(
            loss_g_ae, g_vars, hparams=self._hparams.opt)
        train_op_d = get_train_op(
            loss_d, d_vars, hparams=self._hparams.opt)

        # Interface tensors
        self.predictions = {
            "pred_clas": clas_preds,
            "ground_truth": labels
        }
        self.losses = {
            "loss_g": loss_g,
            "loss_g_ae": loss_g_ae,
            "loss_g_clas": loss_g_clas,
            "loss_d": loss_d_clas
        }
        self.metrics = {
            "accu_d": accu_d,
            "accu_g": accu_g,
            "accu_g_gdy": accu_g_gdy,
        }
        self.train_ops = {
            "train_op_g": train_op_g,
            "train_op_g_ae": train_op_g_ae,
            "train_op_d": train_op_d
        }
        self.samples = {
            "original": inputs['text_ids'][:, 1:],
            "transferred": outputs_.sample_id
        }

        self.fetches_train_g = {
            "loss_g": self.train_ops["train_op_g"],
            "loss_g_ae": self.losses["loss_g_ae"],
            "loss_g_clas": self.losses["loss_g_clas"],
            "accu_g": self.metrics["accu_g"],
            "accu_g_gdy": self.metrics["accu_g_gdy"],
        }
        self.fetches_train_d = {
            "loss_d": self.train_ops["train_op_d"],
            "accu_d": self.metrics["accu_d"]
        }
        fetches_eval = {"batch_size": get_batch_size(inputs['text_ids'])}
        fetches_eval.update(self.losses)
        fetches_eval.update(self.metrics)
        fetches_eval.update(self.samples)
        fetches_eval.update(self.predictions)
        self.fetches_eval = fetches_eval

