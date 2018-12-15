import os
import sys
import numpy as np
import tensorflow as tf

import src.util


class DataPrepper:
    SAMPLES_PER_FILE = 100

    def __init__(self, config, train_data, test_data):
        self.config = config
        self.train = np.array(train_data, dtype=np.uint32)
        self.test = np.array(test_data, dtype=np.uint32)

    def _create_filename(self, dir_name, file_idx):
        filename = "{}/{}.tfrecord".format(dir_name, file_idx)
        return filename

    def _float_feature(self, val):
        if not isinstance(val, list):
            val = [val]
        return tf.train.Feature(float_list=tf.train.FloatList(value=val))

    def _add_sample_to_TFR(self, sample, writer):
        sample = list(sample.astype(dtype=np.float32))
        example = tf.train.Example(features=tf.train.Features(feature={'movie_ratings': self._float_feature(sample)}))
        writer.write(example.SerializeToString())

    def prep(self):
        for dataset, dir_name in zip([self.train, self.test],
                                           [self.config.get_tfr_train_data_dir_loc(),
                                            self.config.get_tfr_test_data_dir_loc()]):
            i = 0
            file_idx = 0
            num_samples = len(dataset)

            while i < num_samples:
                filename = self._create_filename(dir_name, file_idx)

                with tf.python_io.TFRecordWriter(filename) as tfwriter:
                    j = 0

                    while i < num_samples and j < DataPrepper.SAMPLES_PER_FILE:
                        sample = dataset[i]
                        self._add_sample_to_TFR(sample, tfwriter)

                        i += 1
                        j += 1
                    file_idx += 1

    def get_prepped_training_data(self, FLAGS):
        filenames = [FLAGS.tf_records_train_path + filename for filename in os.listdir(FLAGS.tf_records_train_path)]

        train_dataset = tf.data.TFRecordDataset(filenames)
        train_dataset = train_dataset.map(self.parse)
        train_dataset = train_dataset.shuffle(buffer_size=500)
        train_dataset = train_dataset.repeat()
        train_dataset = train_dataset.batch(FLAGS.batch_size)
        train_dataset = train_dataset.prefetch(buffer_size=1)

        val_dataset = tf.data.TFRecordDataset(filenames)
        val_dataset = val_dataset.map(self.parse)
        val_dataset = val_dataset.shuffle(buffer_size=1)
        val_dataset = val_dataset.repeat()
        val_dataset = val_dataset.batch(1)
        val_dataset = val_dataset.prefetch(buffer_size=1)

        return train_dataset, val_dataset

    def get_prepped_testing_data(self, FLAGS):
        filenames = [FLAGS.tf_records_test_path + filename for filename in os.listdir(FLAGS.tf_records_test_path)]

        test_dataset = tf.data.TFRecordDataset(filenames)
        test_dataset = test_dataset.map(self.parse)
        test_dataset = test_dataset.shuffle(buffer_size=1)
        test_dataset = test_dataset.repeat()
        test_dataset = test_dataset.batch(1)
        test_dataset = test_dataset.prefetch(buffer_size=1)

        return test_dataset

    def parse(self, serialized_data):
        features = {'movie_ratings': tf.FixedLenFeature([470758], tf.float32)}

        parsed_example = tf.parse_single_example(serialized_data, features=features)
        movie_ratings = tf.cast(parsed_example['movie_ratings'], tf.float32)

        return movie_ratings
