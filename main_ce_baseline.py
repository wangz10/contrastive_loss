'''
Script to run baseline MLP with cross-entropy loss on 
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

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def parse_option():
    parser = argparse.ArgumentParser('arguments for training baseline MLP')
    # training params
    parser.add_argument('--batch_size', type=int, default=32,
                        help='batch size training'
                        )
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate training'
                        )
    parser.add_argument('--epoch', type=int, default=20,
                        help='Number of epochs for training')

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

    optimizer = tf.keras.optimizers.Adam(lr=args.lr)
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
        (x_train, y_train)).shuffle(5000).batch(args.batch_size)

    test_ds = tf.data.Dataset.from_tensor_slices(
        (x_test, y_test)).batch(args.batch_size)

    # 1. the baseline MLP model
    mlp = MLP(normalize=True, activation=args.activation)
    cce_loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True)

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='train_ACC')

    test_loss = tf.keras.metrics.Mean(name='test_loss')
    test_acc = tf.keras.metrics.SparseCategoricalAccuracy(name='test_ACC')

    @tf.function
    def train_step_baseline(x, y):
        with tf.GradientTape() as tape:
            y_preds = mlp(x, training=True)
            loss = cce_loss_obj(y, y_preds)

        gradients = tape.gradient(loss,
                                  mlp.trainable_variables)
        optimizer.apply_gradients(zip(gradients,
                                      mlp.trainable_variables))

        train_loss(loss)
        train_acc(y, y_preds)

    @tf.function
    def test_step_baseline(x, y):
        y_preds = mlp(x, training=False)
        t_loss = cce_loss_obj(y, y_preds)
        test_loss(t_loss)
        test_acc(y, y_preds)

    model_name = 'baseline'
    if args.write_summary:
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/%s/%s/%s/train' % (
            model_name, args.data, current_time)
        test_log_dir = 'logs/%s/%s/%s/test' % (
            model_name, args.data, current_time)
        train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    for epoch in range(args.epoch):
        # Reset the metrics at the start of the next epoch
        train_loss.reset_states()
        train_acc.reset_states()
        test_loss.reset_states()
        test_acc.reset_states()

        for x, y in train_ds:
            train_step_baseline(x, y)

        if args.write_summary:
            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_acc.result(), step=epoch)

        for x_te, y_te in test_ds:
            test_step_baseline(x_te, y_te)

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

    # get the projections from the last hidden layer before output
    x_tr_proj = mlp.get_last_hidden(x_train)
    x_te_proj = mlp.get_last_hidden(x_test)
    # convert tensor to np.array
    x_tr_proj = x_tr_proj.numpy()
    x_te_proj = x_te_proj.numpy()
    print(x_tr_proj.shape, x_te_proj.shape)
    # 2. Check learned embedding
    if args.draw_figures:
        # do PCA for the projected data
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
        title = 'Data: %s; Embedding: MLP' % args.data
        ax.set_title(title)
        fig.savefig('figs/PCA_plot_%s_MLP_last_layer.png' % args.data)
        # density plot for PCA
        g = sns.jointplot('PC1', 'PC2', data=x_te_proj_pca_df,
                          kind="hex"
                          )
        plt.subplots_adjust(top=0.95)
        g.fig.suptitle(title)
        g.savefig('figs/Joint_PCA_plot_%s_MLP_last_layer.png' % args.data)


if __name__ == '__main__':
    main()
