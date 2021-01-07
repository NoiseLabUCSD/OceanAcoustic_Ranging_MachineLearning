#!/usr/bin/env python
# written by Dr. Haiqiang Niu, Fall 2016

import numpy



class DataSet(object):
    def __init__(self, images, labels, fake_data=False):
        if fake_data:
            self._num_examples = 10000
        else:
            assert images.shape[0] == labels.shape[0], (
                "images.shape: %s labels.shape: %s" % (images.shape,
                                                       labels.shape))
            self._num_examples = images.shape[0]
            # Convert shape from [num examples, rows, columns, depth]
            # to [num examples, rows*columns] (assuming depth == 1)
            #assert images.shape[3] == 1
            #images = images.reshape(images.shape[0],
            #                        images.shape[1] * images.shape[2])
            # Convert from [0, 255] -> [0.0, 1.0].
            #            images = images.astype(numpy.float32)
            #images = numpy.multiply(images, 1.0 / 255.0)
        self._images = images
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        """Return the next `batch_size` examples from this data set."""
        if fake_data:
            fake_image = [1.0 for _ in xrange(34)]
            fake_label = 0
            return [fake_image for _ in xrange(batch_size)], [
                fake_label for _ in xrange(batch_size)]
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            #numpy.random.seed(42)
            perm = numpy.arange(self._num_examples)
            numpy.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        #print(start,end)
        return self._images[start:end], self._labels[start:end]


def read_data_sets(train_dir, FileName,fake_data=False):
    class DataSets(object):
        pass
    data_sets = DataSets()
    if fake_data:
        data_sets.train = DataSet([], [], fake_data=True)
        data_sets.validation = DataSet([], [], fake_data=True)
        data_sets.test = DataSet([], [], fake_data=True)
        return data_sets
    
    train_images = numpy.loadtxt(train_dir + '/train_input/'+ FileName)
    train_labels = numpy.loadtxt(train_dir + '/train_label/'+FileName)
    test_images = numpy.loadtxt(train_dir + '/test_input/'+FileName)
    test_labels = numpy.loadtxt(train_dir + '/test_label/'+FileName)
#    train_images = numpy.loadtxt(train_dir + '/Noise09_traindata_x_450Hz.txt')
#    train_labels = numpy.loadtxt(train_dir + '/Noise09_traindata_y_450Hz.txt')
#    test_images = numpy.loadtxt(train_dir + '/Noise09_testdata_x_450Hz.txt')
#    test_labels = numpy.loadtxt(train_dir + '/Noise09_testdata_y_450Hz.txt')

    if train_labels.ndim == 1:
        train_labels = numpy.reshape(train_labels,(train_labels.size,1))
        test_labels = numpy.reshape(test_labels,(test_labels.size,1))

    print(train_images.shape,train_labels.shape,test_images.shape,test_labels.shape)
    data_sets.train = DataSet(train_images, train_labels)
    data_sets.test = DataSet(test_images, test_labels)
    return data_sets
