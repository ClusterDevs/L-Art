from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys

import tensorflow as tf

from datasets import dataset_utils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
    'dataset_dir',
    None,
    'The directory where the output TFRecords are saved.')

tf.app.flags.DEFINE_boolean(
    'check_image',
    False,
    'Validate the image files only, no processing.')


_PERCENT_VALIDATION = .25


_RANDOM_SEED = 0


_NUM_SHARDS = 5


class ImageReader(object):

    def __init__(self):

        self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
        self._decode_jpeg = tf.image.decode_jpeg(
            self._decode_jpeg_data, channels=3)

    def read_image_dims(self, sess, image_data):
        image = self.decode_jpeg(sess, image_data)
        return image.shape[0], image.shape[1]

    def decode_jpeg(self, sess, image_data):
        image = sess.run(self._decode_jpeg,
                         feed_dict={self._decode_jpeg_data: image_data})
        assert len(image.shape) == 3
        assert image.shape[2] == 3
        return image


def _get_filenames_and_classes(dataset_dir):

    art_root = os.path.join(dataset_dir, 'met_art')
    directories = []
    class_names = []
    for filename in os.listdir(art_root):
        path = os.path.join(art_root, filename)
        if os.path.isdir(path):
            directories.append(path)
            class_names.append(filename)

    photo_filenames = []
    for directory in directories:
        for filename in os.listdir(directory):
            path = os.path.join(directory, filename)
            photo_filenames.append(path)

    return photo_filenames, sorted(class_names)


def _get_dataset_filename(dataset_dir, split_name, shard_id):
    output_filename = 'arts_%s_%05d-of-%05d.tfrecord' % (
        split_name, shard_id, _NUM_SHARDS)
    return os.path.join(dataset_dir, output_filename)


def _check_image(filenames):
 

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:
            for i in range(len(filenames)):
                sys.stdout.write('\r>> Checking image %d/%d' % (
                    i+1, len(filenames)))
                sys.stdout.flush()

                try:

                    image_data = tf.gfile.FastGFile(filenames[i], 'rb').read()
                    height, width = image_reader.read_image_dims(
                        sess, image_data)
                except:
                    sys.stdout.write('\n Error in image:  %s\n' %
                                     (filenames[i]))

    sys.stdout.write('\n')
    sys.stdout.flush()


def _convert_dataset(split_name, filenames, class_names_to_ids, dataset_dir):
 
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(filenames) / float(_NUM_SHARDS)))

    with tf.Graph().as_default():
        image_reader = ImageReader()

        with tf.Session('') as sess:

            for shard_id in range(_NUM_SHARDS):
                output_filename = _get_dataset_filename(
                    dataset_dir, split_name, shard_id)

                with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
                    start_ndx = shard_id * num_per_shard
                    end_ndx = min((shard_id+1) * num_per_shard, len(filenames))
                    for i in range(start_ndx, end_ndx):

                        sys.stdout.write(
                            '>> Converting image %s \n' % (filenames[i]))
                        sys.stdout.flush()

                        image_data = tf.gfile.FastGFile(
                            filenames[i], 'rb').read()
                        height, width = image_reader.read_image_dims(
                            sess, image_data)

                        class_name = os.path.basename(
                            os.path.dirname(filenames[i]))
                        class_id = class_names_to_ids[class_name]

                        example = dataset_utils.image_to_tfexample(
                            image_data, b'jpg', height, width, class_id)
                        tfrecord_writer.write(example.SerializeToString())

    sys.stdout.write('\n')
    sys.stdout.flush()


def _dataset_exists(dataset_dir):
    for split_name in ['train', 'validation']:
        for shard_id in range(_NUM_SHARDS):
            output_filename = _get_dataset_filename(
                dataset_dir, split_name, shard_id)
            if not tf.gfile.Exists(output_filename):
                return False
    return True


def main(_):

    if not FLAGS.dataset_dir:
        raise ValueError(
            'Please specify the dataset directory with --dataset_dir')

    if _dataset_exists(FLAGS.dataset_dir):
        print('Dataset files already exist. Exiting without re-creating them.')
        return

    photo_filenames, class_names = _get_filenames_and_classes(
        FLAGS.dataset_dir)
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    if FLAGS.check_image:
        _check_image(photo_filenames)
        return

    random.seed(_RANDOM_SEED)
    random.shuffle(photo_filenames)
    _NUM_VALIDATION = int(len(photo_filenames) * _PERCENT_VALIDATION)
    training_filenames = photo_filenames[_NUM_VALIDATION:]
    validation_filenames = photo_filenames[:_NUM_VALIDATION]

    _convert_dataset('train', training_filenames, class_names_to_ids,
                     FLAGS.dataset_dir)
    _convert_dataset('validation', validation_filenames, class_names_to_ids,
                     FLAGS.dataset_dir)

    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, FLAGS.dataset_dir)

    print('\nFinished converting the Arts dataset!')


if __name__ == '__main__':
    tf.app.run()
