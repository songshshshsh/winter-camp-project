"""Config
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# pylint: disable=invalid-name

import copy

max_nepochs = 100 # Total number of training epochs
                 # (including pre-train and full-train)
pretrain_nepochs = 10 # Number of pre-train epochs (training as autoencoder)
display = 500  # Display the training results every N training steps.
display_eval = 1e10 # Display the dev results every N training steps (set to a
                    # very large value to disable it).
sample_path = './samples'
checkpoint_path = './checkpoints'
restore = './checkpoints'   # Model snapshot to restore from
result_path = './result'

lambda_g = 0.1    # Weight of the classification loss
gamma_decay = 0.5 # Gumbel-softmax temperature anneal rate

train_data = {
    'batch_size': 1,
    #'seed': 123,
    'datasets': [
        {
            'files': 'train.text',
            'vocab_file': 'vocab',
            'data_name': ''
        },
        {
            'files': 'train.labels0',
            'data_type': 'int',
            'data_name': 'labels0'
        },
        {
            'files': 'train.labels1',
            'data_type': 'int',
            'data_name': 'labels1'
        },
        {
            'files': 'train.labels2',
            'data_type': 'int',
            'data_name': 'labels2'
        },
        {
            'files': 'train.labels3',
            'data_type': 'int',
            'data_name': 'labels3'
        }
    ],
    'name': 'train'
}

val_data = copy.deepcopy(train_data)
val_data['datasets'][0]['files'] = 'val.text'
val_data['datasets'][1]['files'] = 'val.labels0'
val_data['datasets'][2]['files'] = 'val.labels1'
val_data['datasets'][3]['files'] = 'val.labels2'
val_data['datasets'][4]['files'] = 'val.labels3'

test_data = copy.deepcopy(train_data)
test_data['datasets'][0]['files'] = 'small_test.text'
test_data['datasets'][1]['files'] = 'small_test.labels0'
test_data['datasets'][2]['files'] = 'small_test.labels1'
test_data['datasets'][3]['files'] = 'small_test.labels2'
test_data['datasets'][4]['files'] = 'small_test.labels3'

eval_data = copy.deepcopy(train_data)
eval_data['datasets'][0]['files'] = 'eval.text'
eval_data['datasets'][1]['files'] = 'eval.labels0'
eval_data['datasets'][2]['files'] = 'eval.labels1'
eval_data['datasets'][3]['files'] = 'eval.labels2'
eval_data['datasets'][4]['files'] = 'eval.labels3'

model = {
    'dim_c': 200,
    'dim_z': 500,
    'embedder': {
        'dim': 100,
    },
    'encoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700
            },
            'dropout': {
                'input_keep_prob': 0.5
            }
        }
    },
    'decoder': {
        'rnn_cell': {
            'type': 'GRUCell',
            'kwargs': {
                'num_units': 700,
            },
            'dropout': {
                'input_keep_prob': 0.5,
                'output_keep_prob': 0.5
            },
        },
        'attention': {
            'type': 'BahdanauAttention',
            'kwargs': {
                'num_units': 700,
            },
            'attention_layer_size': 700,
        },
        'max_decoding_length_train': 25,
        'max_decoding_length_infer': 25,
    },
    'classifier': {
        'kernel_size': [3, 4, 5],
        'filters': 128,
        'other_conv_kwargs': {'padding': 'same'},
        'dropout_conv': [1],
        'dropout_rate': 0.5,
        'num_dense_layers': 0,
        'num_classes': 1
    },
    'opt': {
        'optimizer': {
            'type':  'AdamOptimizer',
            'kwargs': {
                'learning_rate': 5e-4,
            },
        },
    },
}
