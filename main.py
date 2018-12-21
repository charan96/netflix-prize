import os
import sys
import time
import tensorflow as tf
from collections import OrderedDict

from src.util import Util
from src.config_loader import Config
from src.data_matrix import DataMatrix
from src.data_prepper import DataPrepper
from src.autoencoder import Autoencoder

# tf.enable_eager_execution()

start = time.time()

CONFIG_FILE = "./config.json"

config = Config(CONFIG_FILE)


def set_flags(epochs=20,
              batch_size=64,
              lr=0.005,
              l2_reg=True,
              decay=0.01,
              num_movies=470758,
              hidden_neurons=4096,
              num_users=4499):
    tf.app.flags.DEFINE_string('tf_records_train_path', config.get_tfr_train_data_dir_loc(), 'Path of the training data.')

    tf.app.flags.DEFINE_string('tf_records_test_path', config.get_tfr_test_data_dir_loc(), 'Path of the test data.')

    tf.app.flags.DEFINE_integer('num_epoch', epochs, 'Number of training epochs.')
    tf.app.flags.DEFINE_integer('batch_size', batch_size, 'Size of the training batch.')
    tf.app.flags.DEFINE_float('learning_rate', lr, 'Learning_Rate')
    tf.app.flags.DEFINE_boolean('l2_reg', l2_reg, 'L2 regularization.')
    tf.app.flags.DEFINE_float('lambda_', decay, 'Wight decay factor.')
    tf.app.flags.DEFINE_integer('num_v', num_movies, 'Number of visible neurons (Number of movies the users rated.)')
    tf.app.flags.DEFINE_integer('num_h', hidden_neurons, 'Number of hidden neurons.)')
    tf.app.flags.DEFINE_integer('num_samples', num_users,
                                'Number of training samples (Number of users, who gave a rating).')
    flags = tf.app.flags.FLAGS

    print('learning_rate: {}'.format(lr))
    print('batch: {}'.format(batch_size))
    print('reg: {}'.format(l2_reg))
    return flags

FLAGS = set_flags()

# ====================================================================

# BUILDING THE DATA MATRIX
dmatrix_obj = DataMatrix(config)
# dmatrix_obj.build_matrix_from_scratch()
# dmatrix_obj.train_test_split_matrix(use_saved=False)

# LOADING PRE-BUILT DATA MATRIX
dmatrix_obj.load_train_matrix()
dmatrix_obj.load_test_matrix()

# DATA PREP
dprep = DataPrepper(config, dmatrix_obj.train, dmatrix_obj.test)
# dprep.prep()


num_batches = int(FLAGS.num_samples / FLAGS.batch_size)

train_data, val_data = dprep.get_prepped_training_data(FLAGS=FLAGS)
test_data = dprep.get_prepped_testing_data(FLAGS=FLAGS)
print('got prepped training and testing data')

train_iter = train_data.make_initializable_iterator()
val_iter = val_data.make_initializable_iterator()
test_iter = test_data.make_initializable_iterator()
print('made iterators')

train_x = train_iter.get_next()
val_x = val_iter.get_next()
test_x = test_iter.get_next()
print('got next for all iterators')

model = Autoencoder(FLAGS=FLAGS)
print('made model')

train_op, train_loss_op = model._optimizer(train_x)
pred_op, test_loss_op = model._validation_loss(val_x, test_x)


print('starting session')
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_loss = 0
    test_loss = 0

    for epoch in range(FLAGS.num_epoch):
        sess.run(train_iter.initializer)

        for batch_nr in range(num_batches):
            _, loss_ = sess.run((train_op, train_loss_op))
            train_loss += loss_

        sess.run(val_iter.initializer)
        sess.run(test_iter.initializer)

        for i in range(FLAGS.num_samples):
            pred, loss_ = sess.run((pred_op, test_loss_op))
            test_loss += loss_

        print('epoch_nr: %i, train_loss: %.3f, test_loss: %.3f' % (
            epoch, (train_loss / num_batches), (test_loss / FLAGS.num_samples)))
        
        train_loss = 0
        test_loss = 0

# ====================================================================
scratch_dict = OrderedDict()

# Util.write_data_to_scratch_file(scratch_dict)

print(time.time() - start)
