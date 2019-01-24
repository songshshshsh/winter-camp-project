# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name, too-many-locals, too-many-arguments, no-member

import os
import sys
import importlib
import argparse
import codecs
import numpy as np
import tensorflow as tf
import texar as tx

sys.path.append(os.path.abspath('../'))
os.chdir('..')
from ctrl_gen_model_for_val import CtrlGenModelVal

config = importlib.import_module('eval_config')
print(config)

def label_to_list(label):
	return [(int)(label[0] == 'I'), (int)(label[1] == 'N'), (int)(label[2] == 'T'), (int)(label[3] == 'J')]

def eval(text, label0 = None, label1 = None, mode = 'classification'):
    # Data
    eval_data_path = config.eval_data['datasets'][0]['files']
    origin_code = label_to_list(label0)
    for i in range(4):
        #gg = str(bytes(str(origin_code[i]) + u'\n', encoding = 'utf-8'), encoding = 'utf-8')
        gg = str(origin_code[i]) + '\n'
        print(type(gg))
        codecs.open(config.eval_data['datasets'][i+1]['files'], 'wb', 'utf-8', 'ignore').write(gg)
        # codecs.open(config.eval_data['datasets'][i+1]['files'], 'wb', 'utf-8', 'ignore').write(gg)
    #strr = str(bytes(text + u'\n', encoding = 'utf-8'), encoding = 'utf-8')
    strr = str(text + '\n')
    f = codecs.open(eval_data_path, 'wb', 'utf-8', 'ignore')
    f.write(strr)
    f.close()
    eval_data = tx.data.MultiAlignedData(config.eval_data)
    #eval_data = tx.data.MultiAlignedData(config.train_data)
    code = 1 - np.array(label_to_list(label1))
    #print(code, origin_code, text)
    sys.stdout.flush()
    print(233)
    print(233)
    print(233)
    print(233)
    print(233)
    print(233)
    print(233)
    print(233)
    vocab = eval_data.vocab(0)

    # Each training batch is used twice: once for updating the generator and
    # once for updating the discriminator. Feedable data iterator is used for
    # such case.
    iterator = tx.data.FeedableDataIterator({'test': eval_data})
    batch = iterator.get_next()

    # Model
    gamma = tf.placeholder(dtype=tf.float32, shape=[], name='gamma')
    lambda_g = tf.placeholder(dtype=tf.float32, shape=[], name='lambda_g')
    model = CtrlGenModelVal(batch, vocab, gamma, lambda_g, code, config.model)

    def _eval_epoch(sess, gamma_, lambda_g_, val_or_test='val'):
        avg_meters = tx.utils.AverageRecorder()

        while True:
            try:
                feed_dict = {
                    iterator.handle: iterator.get_handle(sess, val_or_test),
                    gamma: gamma_,
                    lambda_g: lambda_g_,
                    tx.context.global_mode(): tf.estimator.ModeKeys.EVAL
                }
                #print(sess.run(model.fe, feed_dict=feed_dict))

                vals = sess.run(model.fetches_eval, feed_dict=feed_dict)
                print(vals)
                batch_size = vals.pop('batch_size')
                predictions = vals.pop('predictions')
                ground_truth = vals.pop('ground_truth')

                # Compute  BLEU
                samples = tx.utils.dict_pop(vals, list(model.samples.keys()))
                hyps = tx.utils.map_ids_to_strs(samples['transferred'], vocab)

                refs = tx.utils.map_ids_to_strs(samples['original'], vocab)
                refs = np.expand_dims(refs, axis=1)

                bleu = tx.evals.corpus_bleu_moses(refs, hyps)
                vals['bleu'] = bleu

                avg_meters.add(vals, weight=batch_size)

                # Writes samples
                refs = refs.squeeze().reshape((-1))
                print('predictions', predictions, ground_truth)
                if mode == 'classification':
                    return predictions
                else:
                    return refs[0]

            except tf.errors.OutOfRangeError:
                print('{}: {}'.format(
                    val_or_test, avg_meters.to_str(precision=4)))
                break

        return avg_meters.avg()

    # tf.gfile.MakeDirs(config.sample_path)
    # tf.gfile.MakeDirs(config.checkpoint_path)
    tf.gfile.MakeDirs(config.result_path)

    # Runs the logics
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    sess.run(tf.tables_initializer())

    saver = tf.train.Saver(max_to_keep=None)
    if config.restore:
        print('Restore from: {}'.format(config.restore))
        saver.restore(sess, tf.train.latest_checkpoint(config.restore))

    iterator.initialize_dataset(sess)

    gamma_ = 1.
    lambda_g_ = 0.
    gamma_ = max(0.001, gamma_ * config.gamma_decay)
    lambda_g_ = config.lambda_g
    iterator.restart_dataset(sess, 'test')
    result = _eval_epoch(sess, gamma_, lambda_g_, 'test')
    tf.reset_default_graph()
    return result


def getClass(text):
    class_label = eval(text, 'INTJ', 'INTJ', 'classification')
    class_name = ''
    print(class_label)
    if class_label[0][0] == 0:
        class_name += 'I'
    else:
        class_name += 'E'
    if class_label[0][1] == 0:
        class_name += 'N'
    else:
        class_name += 'S'
    if class_label[0][1] == 0:
        class_name += 'T'
    else:
        class_name += 'F'
    if class_label[0][1] == 0:
        class_name += 'J'
    else:
        class_name += 'P'
    return class_name

def getToken(text, label0, label1):
    generation = eval(text, label0, label1, 'generation')
    return generation
