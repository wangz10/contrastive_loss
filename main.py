'''
Script to run various two-stage supervised contrastive loss functions on 
MNIST or Fashion MNIST data.

Author: Zichen Wang (wangzc921@gmail.com)
'''
import argparse
import datetime
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

from model import *
import losses

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

LOSS_NAMES = {
    'max_margin': 'Max margin contrastive',
    'npairs': 'Multiclass N-pairs',
    'sup_nt_xent': 'Supervised NT-Xent',
    'triplet-hard': 'Triplet hard',
    'triplet-semihard': 'Triplet semihard',
    'triplet-soft': 'Triplet soft'
}


def parse_option():
    parser = argparse.ArgumentParser('arguments for two-stage training ')
    # training params
    parser.add_argument('--batch_size_1', type=int, default=512,
                        help='batch size for stage 1 pretraining'
                        )
    parser.add_argument('--batch_size_2', type=int, default=32,
                        help='batch size for stage 2 training'
                        )
    parser.add_argument('--lr_1', type=float, default=0.5,
                        help='learning rate for stage 1 pretraining'
                        )
    parser.add_argument('--lr_2', type=float, default=0.001,
                        help='learning rate for stage 2 training'
                        )
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs for training in stage1, the same number of epochs will be applied on stage2')
    parser.add_argument('--optimizer', type=str, default='adam',
                        help='Optimizer to use, choose from ("adam", "lars", "sgd")'
                        )
    # loss functions
    parser.add_argument('--loss', type=str, default='max_margin',
                        help='Loss function used for stage 1, choose from ("max_margin", "npairs", "sup_nt_xent", "triplet-hard", "triplet-semihard", "triplet-soft")')
    parser.add_argument('--margin', type=float, default=1.0,
                        help='margin for tfa.losses.contrastive_loss. will only be used when --loss=max_margin')
    parser.add_argument('--metric', type=str, default='euclidean',
                        help='distance metrics for tfa.losses.contrastive_loss, choose from ("euclidean", "cosine"). will only be used when --loss=max_margin')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='temperature for sup_nt_xent loss. will only be used when --loss=sup_nt_xent')
    parser.add_argument('--base_temperature', type=float, default=0.07,
                        help='base_temperature for sup_nt_xent loss. will only be used when --loss=sup_nt_xent')
    # dataset params
    parser.add_argument('--data', type=str, default='mnist',
                        help='Dataset to choose from ("mnist", "fashion_mnist")'
                        )
    parser.add_argument('--n_data_train', type=int, default=60000,
                        help='number of data points used for training both stage 1 and 2'
                        )

    # model architecture
    parser.add_argument('--projection_dim', type=int, default=128,
                        help='output tensor dimension from projector'
                        )
    parser.add_argument('--activation', type=str, default='leaky_relu',
                        help='activation function between hidden layers'
                        )

    # output options
    parser.add_argument('--write_summary', action='store_true',
                        help='write summary for tensorboard'
                        )
    parser.add_argument('--draw_figures', action='store_true',
                        help='produce figures for the projections'
                        )

    args = parser.parse_args()
    return args


def main():
    args = parse_option()
    print(args)

    # check args
    if args.loss not in LOSS_NAMES:
        raise ValueError('Unsupported loss function type {}'.format(args.loss))

    if args.optimizer == 'adam':
        optimizer1 = tf.keras.optimizers.Adam(lr=args.lr_1)
    elif args.optimizer == 'lars':
        from lars_optimizer import LARSOptimizer
        # not compatible with tf2
        optimizer1 = LARSOptimizer(args.lr_1,
                                   exclude_from_weight_decay=['batch_normalization', 'bias'])
    elif args.optimizer == 'sgd':
        optimizer1 = tfa.optimizers.SGDW(learning_rate=args.lr_1,
                                         momentum=0.9,
                                         weight_decay=1e-4
                                         )
    optimizer2 = tf.keras.optimizers.Adam(lr=args.lr_2)

    model_name = '{}_model-bs_{}-lr_{}'.format(
        args.loss, args.batch_size_1, args.lr_1)

    # 0. Load data
    if args.data == 'mnist':
        mnist = tf.keras.datasets.mnist
    elif args.data == 'fashion_mnist':
        mnist = tf.keras.datasets.fashion_mnist
    print('Loading {} data...'.format(args.data))
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0
    x_train = x_train.reshape(-1, 28*28).astype(np.float32)
    x_test = x_test.reshape(-1, 28*28).astype(np.float32)
    print(x_train.shape, x_test.shape)

    # simulate low data regime for training
    n_train = x_train.shape[0]
    shuffle_idx = np.arange(n_train)
    np.random.shuffle(shuffle_idx)

    x_train = x_train[shuffle_idx][:args.n_data_train]
    y_train = y_train[shuffle_idx][:args.n_data_train]
    print('Training dataset shapes after slicing:')
    print(x_train.shape, y_train.shape)

    train_ds = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(5000).batch(args.batch_size_1)

    train_ds2 = tf.data.Dataset.from_tensor_slices(
        (x_train, y_train)).shuffle(5000).batch(args.batch_size_1)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(args.batch_size_1)

    # 1. Stage 1: train encoder with multiclass N-pair loss
    encoder = Encoder(normalize=True, activation=args.activation)
    projector = Projector(args.projection_dim,
                          normalize=True, activation=args.activation)

    if args.loss == 'max_margin':
        def loss_func(z, y): return losses.max_margin_contrastive_loss(
            z, y, margin=args.margin, metric=args.metric)
    elif args.loss == 'npairs':
        loss_func = losses.multiclass_npairs_loss
    elif args.loss == 'sup_nt_xent':
        def loss_func(z, y): return losses.supervised_nt_xent_loss(
            z, y, temperature=args.temperature, base_temperature=args.base_temperature)
    elif args.loss.startswith('triplet'):
        triplet_kind = args.loss.split('-')[1]
        def loss_func(z, y): return losses.triplet_loss(
            z, y, kind=triplet_kind, margin=args.margin)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    test_loss = tf.keras.metrics.Mean(name='test_loss')

    # tf.config.experimental_run_functions_eagerly(True)
    @tf.function
    # train step for the contrastive loss
    def train_step_stage1(x, y):
        '''
        x: data tensor, shape: (batch_size, data_dim)
        y: data labels, shape: (batch_size, )
        '''
        with tf.GradientTape() as tape:
            r = encoder(x, training=True)
            z = projector(r, training=True)
            loss = loss_func(z, y)

        gradients = tape.gradient(loss,
                                  encoder.trainable_variables + projector.trainable_variables)
        optimizer1.apply_gradients(zip(gradients,
                                       encoder.trainable_variables + projector.trainable_variables))
        train_loss(loss)

    @tf.function
    def test_step_stage1(x, y):
        r = encoder(x, training=False)
        z = projector(r, training=False)
        t_loss = loss_func(z, y)
        test_loss(t_loss)

    print('Stage 1 training ...')
    for epoch in range(args.epoch):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        test_loss.reset_states()

        for x, y in train_ds:
            train_step_stage1(x, y)

        for x_te, y_te in test_ds:
            test_step_stage1(x_te, y_te)

        template = 'Epoch {}, Loss: {}, Test Loss: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              test_loss.result()))

    if args.draw_figures:
        # projecting data with the trained encoder, projector
        x_tr_proj = projector(encoder(x_train))
        x_te_proj = projector(encoder(x_test))
        # convert tensor to np.array
        x_tr_proj = x_tr_proj.numpy()
        x_te_proj = x_te_proj.numpy()
        print(x_tr_proj.shape, x_te_proj.shape)

        # check learned embedding using PCA
        pca = PCA(n_components=2)
        pca.fit(x_tr_proj)
        x_te_proj_pca = pca.transform(x_te_proj)

        x_te_proj_pca_df = pd.DataFrame(x_te_proj_pca, columns=['PC1', 'PC2'])
        x_te_proj_pca_df['label'] = y_test
        # PCA scatter plot
        fig, ax = plt.subplots()
        ax = sns.scatterplot('PC1', 'PC2',
                             data=x_te_proj_pca_df,
                             palette='tab10',
                             hue='label',
                             linewidth=0,
                             alpha=0.6,
                             ax=ax
                             )

        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        title = 'Data: {}\nEmbedding: {}\nbatch size: {}; LR: {}'.format(
            args.data, LOSS_NAMES[args.loss], args.batch_size_1, args.lr_1)
        ax.set_title(title)
        fig.savefig(
            'figs/PCA_plot_{}_{}_embed.png'.format(args.data, model_name))

        # density plot for PCA
        g = sns.jointplot('PC1', 'PC2', data=x_te_proj_pca_df,
                          kind="hex"
                          )
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(title)

        g.savefig(
            'figs/Joint_PCA_plot_{}_{}_embed.png'.format(args.data, model_name))

    # Stage 2: freeze the learned representations and then learn a classifier
    # on a linear layer using a softmax loss
    softmax = SoftmaxPred()

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_ACC')
    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_ACC')

    cce_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    @tf.function
    # train step for the 2nd stage
    def train_step(x, y):
        '''
        x: data tensor, shape: (batch_size, data_dim)
        y: data labels, shape: (batch_size, )
        '''
        with tf.GradientTape() as tape:
            r = encoder(x, training=False)
            y_preds = softmax(r, training=True)
            loss = cce_loss_obj(y, y_preds)

        # freeze the encoder, only train the softmax layer
        gradients = tape.gradient(loss,
                                  softmax.trainable_variables)
        optimizer2.apply_gradients(zip(gradients,
                                       softmax.trainable_variables))
        train_loss(loss)
        train_acc(y, y_preds)

    @tf.function
    def test_step(x, y):
        r = encoder(x, training=False)
        y_preds = softmax(r, training=False)
        t_loss = cce_loss_obj(y, y_preds)
        test_loss(t_loss)
        test_acc(y, y_preds)

    if args.write_summary:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/{}/{}/{}/train'.format(
            model_name, args.data, current_time)
        test_log_dir = 'logs/{}/{}/{}/test'.format(
            model_name, args.data, current_time)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    print('Stage 2 training ...')
    for epoch in range(args.epoch):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        for x, y in train_ds2:
            train_step(x, y)

        if args.write_summary:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

        for x_te, y_te in test_ds:
            test_step(x_te, y_te)

        if args.write_summary:
            with test_summary_writer.as_default():
                tf.summary.scalar('loss', test_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', test_acc.result(), step=epoch)

        template = 'Epoch {}, Loss: {}, Acc: {}, Test Loss: {}, Test Acc: {}'
        print(template.format(epoch + 1,
                              train_loss.result(),
                              train_acc.result() * 100,
                              test_loss.result(),
                              test_acc.result() * 100))


if __name__ == '__main__':
    main()
