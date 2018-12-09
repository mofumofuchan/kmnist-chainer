#!/usr/bin/env python
import argparse

import numpy as np
import chainer
import chainer.links as L
from chainer import training
from chainer.training import extensions
from chainer.datasets import TupleDataset

import train_kmnist


def main():
    # This script is almost identical to train_mnist.py. The only difference is
    # that this script uses data-parallel computation on two GPUs.
    # See train_mnist.py for more details.
    parser = argparse.ArgumentParser(description='Chainer example: MNIST')
    parser.add_argument('--batchsize', '-b', type=int, default=400,
                        help='Number of images in each mini-batch')
    parser.add_argument('--epoch', '-e', type=int, default=20,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--gpu0', '-g', type=int, default=0,
                        help='First GPU ID')
    parser.add_argument('--gpu1', '-G', type=int, default=1,
                        help='Second GPU ID')
    parser.add_argument('--out', '-o', default='result_parallel',
                        help='Directory to output the result')
    parser.add_argument('--resume', '-r', default='',
                        help='Resume the training from snapshot')
    parser.add_argument('--unit', '-u', type=int, default=1000,
                        help='Number of units')
    parser.add_argument('--train_imgs',
                        default='data/kmnist-train-imgs.npz',
                        help='Path to kmnist training images')
    parser.add_argument('--train_label',
                        default='data/kmnist-train-labels.npz',
                        help='Path to kmnist training labels')
    parser.add_argument('--test_imgs',
                        default='data/kmnist-test-imgs.npz',
                        help='Path to kmnist test images')
    parser.add_argument('--test_label',
                        default='data/kmnist-test-labels.npz',
                        help='Path to kmnist test labels')

    args = parser.parse_args()

    print('GPU: {}, {}'.format(args.gpu0, args.gpu1))
    print('# unit: {}'.format(args.unit))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    chainer.backends.cuda.get_device_from_id(args.gpu0).use()

    model = L.Classifier(train_kmnist.MLP(args.unit, 10))
    optimizer = chainer.optimizers.Adam()
    optimizer.setup(model)

    # Load and prepare the KMNIST dataset
    train_data = np.load(args.train_imgs)['arr_0'].\
                 reshape((60000, 784)).astype(np.float32)/255.
    train_labels = [int(n) for n in np.load(args.train_label)['arr_0']]
    train = TupleDataset(train_data, train_labels)

    test_data = np.load(args.test_imgs)['arr_0'].\
                reshape((10000, 784)).astype(np.float32)/255.
    test_labels = [int(n) for n in np.load(args.test_label)['arr_0']]
    test = TupleDataset(test_data, test_labels)

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    test_iter = chainer.iterators.SerialIterator(test, args.batchsize,
                                                 repeat=False, shuffle=False)

    # ParallelUpdater implements the data-parallel gradient computation on
    # multiple GPUs. It accepts "devices" argument that specifies which GPU to
    # use.
    updater = training.updaters.ParallelUpdater(
        train_iter,
        optimizer,
        # The device of the name 'main' is used as a "master", while others are
        # used as slaves. Names other than 'main' are arbitrary.
        devices={'main': args.gpu0, 'second': args.gpu1},
    )
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)

    trainer.extend(extensions.Evaluator(test_iter, model, device=args.gpu0))
    trainer.extend(extensions.dump_graph('main/loss'))
    trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.ProgressBar())

    if args.resume:
        chainer.serializers.load_npz(args.resume, trainer)

    trainer.run()


if __name__ == '__main__':
    main()
