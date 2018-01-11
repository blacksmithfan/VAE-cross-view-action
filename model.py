import tensorflow as tf
import numpy as np
import sys
from network import *
from tensorflow.python.ops import rnn, rnn_cell
# import tflearn
from tflearn.layers.recurrent import bidirectional_rnn, BasicLSTMCell
from tensorflow.contrib import layers


class Model:
    @staticmethod
    def alexnet(_X, _dropout):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        _X = tf.reshape(_X, [-1, 112, 112, 3])
        _X = tf.image.resize_images(_X, [227, 227])
        with tf.device('/gpu:0'):
            conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
            norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
            # Layer 2 (conv-relu-pool-lrn)
        # with tf.device('/gpu:1'):
            conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
            conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
            norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
            # Layer 3 (conv-relu)
        # with tf.device('/gpu:2'):
            conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
            # Layer 4 (conv-relu)
            conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
            # Layer 5 (conv-relu-pool)
        # with tf.device('/gpu:3'):
            conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
            # Layer 6 (fc-relu-drop)
            fc6 = tf.reshape(pool5, [-1, 6*6*256])
            fc6 = fc(fc6, 6*6*256, 4096, name='fc6')
            fc6 = dropout(fc6, _dropout)
            # Layer 7 (fc-relu-drop)
            fc7 = fc(fc6, 4096, 4096, name='fc7')
            fc7 = dropout(fc7, _dropout)
            # Layer 8 (fc-prob)
            # fc8 = fc(fc7, 4096, n_class, relu=False, name='fc8')
            #
            # x = tf.reshape(fc8, [-1, 20, n_class])
            # softmax_output = tf.reduce_mean(x, 1)
        return fc7


    @staticmethod
    def slimNet(_X, _dropout):

        net = tf.reshape(_X, [-1, 112, 112, 3])
        net = layers.conv2d(net, 32, 5, stride=4)
        net = layers.conv2d(net, 64, 5, stride=4)
        net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
        net = layers.dropout(net, keep_prob=0.9)
        # net = layers.flatten(net)
        net = tf.reshape(net, [-1, 20 * 2 * 2 * 128])

        net = layers.fully_connected(net, 128, activation_fn=None)
        # net = tf.reshape(net, [-1, 2 * 2 *])

        return net

    @staticmethod
    def slimNet_lstm(_X, _dropout):

        net = tf.reshape(_X, [-1, 112, 112, 3])
        net = layers.conv2d(net, 32, 5, stride=4)
        net = layers.conv2d(net, 64, 5, stride=4)
        net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
        net = layers.dropout(net, keep_prob=0.9)
        # net = layers.flatten(net)
        net = tf.reshape(net, [-1, 20, 2 * 2 * 128])

        #
        # net = layers.fully_connected(net, 128, activation_fn=None)
        # net = tf.reshape(net, [-1, 2 * 2 *])

        # x = tf.reshape(fc6, [-1, n_steps, 512])
        net = tf.transpose(net, [1, 0, 2])
        #
        net = tf.reshape(net, [-1, 2 * 2 * 128])
        # print(x.get_shape())
        net = tf.split(0, 20, net)

        lstm_cell = rnn_cell.BasicLSTMCell(128, forget_bias=1.0, state_is_tuple=True)
        #
        outputs, states = rnn.rnn(lstm_cell, net, dtype=tf.float32)
        outputs = tf.reshape(outputs, [-1, 120, 128])
        outputs = tf.transpose(outputs, [1, 0, 2])
        outputs = tf.reshape(outputs, [-1, 20 * 128])

        return outputs

    @staticmethod
    def alexnet_encoder(_X, _dropout, n_class, n_hidden):
        # TODO weight decay loss tern
        # Layer 1 (conv-relu-pool-lrn)
        _X = tf.reshape(_X, [-1, 112, 112, 3])
        _X = tf.image.resize_images(_X, [227, 227])
        with tf.device('/gpu:0'):
            conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
            conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
            norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
            # Layer 2 (conv-relu-pool-lrn)
            # with tf.device('/gpu:1'):
            conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
            conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
            norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
            # Layer 3 (conv-relu)
            # with tf.device('/gpu:2'):
            conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
            # Layer 4 (conv-relu)
            conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
            # Layer 5 (conv-relu-pool)
            # with tf.device('/gpu:3'):
            conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
            pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
            # Layer 6 (fc-relu-drop)
            fc6 = tf.reshape(pool5, [-1, 6 * 6 * 256])
            fc6 = fc(fc6, 6 * 6 * 256, 4096, name='fc6')
            fc6 = dropout(fc6, _dropout)
            # Layer 7 (fc-relu-drop)
            fc7 = fc(fc6, 4096, 4096, name='fc7')
            fc7 = dropout(fc7, _dropout)
            # Layer 8 (fc-prob)
            fc8 = fc(fc7, 4096, n_class, relu=False, name='fc8')

            fc_output = fc(fc7, 4096, n_hidden, relu=False, name='fc9')

            x = tf.reshape(fc8, [-1, 20, n_class])
            softmax_output = tf.reduce_mean(x, 1)
        return softmax_output, fc_output

    @staticmethod
    def alex_decoder(input_tensor):
        '''Create decoder network.

            If input tensor is provided then decodes it, otherwise samples from
            a sampled vector.
        Args:
            input_tensor: a batch of vectors to decode

        Returns:
            A tensor that expresses the decoder network
        '''
        # tf.concat_v2([input_tensor, input_tensor], 0)
        net = tf.expand_dims(input_tensor, 1)  # input (128, 128)
        net = tf.expand_dims(net, 1)  # input (128, 1, 128)
        net = layers.conv2d_transpose(net, 128, 3, padding='VALID')  # input (128, 1, 1, 128)
        net = layers.conv2d_transpose(net, 64, 5, padding='VALID')  # input (128, 3, 3, 128)
        # net = layers.conv2d_transpose(net, 48, 8, padding='VALID')
        net = layers.conv2d_transpose(net, 32, 5, stride=4)  # input (128, 7, 7, 64)
        net = layers.conv2d_transpose(
            net, 3, 5, stride=4, activation_fn=tf.nn.sigmoid)  # input (128, 14, 14, 32), output (128, 28, 28, 1)
        # print(tf.shape(net))
        reformatted_tensor = tf.reshape(net, [-1, 112, 112, 3])
        resized_img = tf.image.resize_images(reformatted_tensor, [227, 227])
        net_flat = layers.flatten(resized_img)
        return net_flat, net


    @staticmethod
    def get_vae_cost(mean, stddev, epsilon=1e-8):
        '''VAE loss
            See the paper
        Args:
            mean:
            stddev:
            epsilon:
        '''
        return tf.reduce_sum(0.5 * (tf.square(mean) + tf.square(stddev) -
                                        2.0 * tf.log(stddev + epsilon) - 1.0))


    @staticmethod
    def alexnet_lstm(_X, _dropout, n_steps, n_hidden, batch_size):
        conv1 = conv(_X, 11, 11, 96, 4, 4, padding='VALID', name='conv1')
        conv1 = max_pool(conv1, 3, 3, 2, 2, padding='VALID', name='pool1')
        norm1 = lrn(conv1, 2, 2e-05, 0.75, name='norm1')
    # Layer 2 (conv-relu-pool-lrn)
        conv2 = conv(norm1, 5, 5, 256, 1, 1, group=2, name='conv2')
        conv2 = max_pool(conv2, 3, 3, 2, 2, padding='VALID', name='pool2')
        norm2 = lrn(conv2, 2, 2e-05, 0.75, name='norm2')
    # Layer 3 (conv-relu)
        conv3 = conv(norm2, 3, 3, 384, 1, 1, name='conv3')
        # Layer 4 (conv-relu)
        conv4 = conv(conv3, 3, 3, 384, 1, 1, group=2, name='conv4')
        # Layer 5 (conv-relu-pool)
        conv5 = conv(conv4, 3, 3, 256, 1, 1, group=2, name='conv5')
        pool5 = max_pool(conv5, 3, 3, 2, 2, padding='VALID', name='pool5')
        # Layer 6 (fc-relu-drop)
        fc6 = tf.reshape(pool5, [-1, 6*6*256])
        fc6 = fc(fc6, 6*6*256, 512, name='fc6')
        fc6 = dropout(fc6, _dropout)
        # Layer 7 (fc-relu-drop)
        # fc7 = fc(fc6, 4096, 4096, name='fc7')
        # fc7 = dropout(fc7, _dropout)
        # Layer 8 (fc-prob)

        x = tf.reshape(fc6, [-1, n_steps, 512])
        x = tf.transpose(x, [1, 0, 2])

        x = tf.reshape(x, [-1, 512])
        print(x.get_shape())
        x = tf.split(0, n_steps, x)

        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)

        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        outputs = dropout(outputs, _dropout)
        return outputs


    @staticmethod
    def decoder(input_tensor):
        '''Create decoder network.

            If input tensor is provided then decodes it, otherwise samples from
            a sampled vector.
        Args:
            input_tensor: a batch of vectors to decode

        Returns:
            A tensor that expresses the decoder network
        '''
        # tf.concat_v2([input_tensor, input_tensor], 0)
        net = tf.expand_dims(input_tensor, 1)  # input (128, 128)
        net = tf.expand_dims(net, 1)  # input (128, 1, 128)
        net = layers.conv2d_transpose(net, 128, 3, padding='VALID')  # input (128, 1, 1, 128)
        net = layers.conv2d_transpose(net, 64, 5, padding='VALID')  # input (128, 3, 3, 128)
        # net = layers.conv2d_transpose(net, 48, 8, padding='VALID')
        net = layers.conv2d_transpose(net, 32, 5, stride=4)  # input (128, 7, 7, 64)
        net = layers.conv2d_transpose(
            net, 3, 5, stride=4, activation_fn=tf.nn.sigmoid)  # input (128, 14, 14, 32), output (128, 28, 28, 1)
        # print(tf.shape(net))
        net_flat = layers.flatten(net)
        return net_flat, net

    @staticmethod
    def get_reconstruction_cost(output_tensor, target_tensor, epsilon=1e-8):
        '''Reconstruction loss

        Cross entropy reconstruction loss

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        # return target_tensor
        return tf.nn.l2_loss(output_tensor-target_tensor)/112
        # return tf.reduce_sum(tf.clip_by_value(tf.abs(output_tensor - target_tensor), 1e-10, 1000000)) / 14884

    @staticmethod
    def get_reconstruction_cost_feature(output_tensor, target_tensor, epsilon=1e-8):
        '''Reconstruction loss

        Cross entropy reconstruction loss

        Args:
            output_tensor: tensor produces by decoder
            target_tensor: the target tensor that we want to reconstruct
            epsilon:
        '''
        # return target_tensor
        return tf.nn.l2_loss(output_tensor-target_tensor)


    @staticmethod
    def encoder(input_tensor, output_size, class_num, keep_rate):

        net = tf.reshape(input_tensor, [-1, 112, 112, 3])
        net = layers.conv2d(net, 32, 5, stride=4)
        net = layers.conv2d(net, 64, 5, stride=4)
        net = layers.conv2d(net, 128, 5, stride=2, padding='VALID')
        net = layers.dropout(net, keep_prob=keep_rate)
        net = layers.flatten(net)
        net = layers.fully_connected(net, output_size, activation_fn=None)

        softmax = layers.fully_connected(net, class_num, activation_fn=None) # Softmax predictions for each video frame
        x = tf.reshape(softmax, [-1, 20, class_num])
        softmax_output = tf.reduce_mean(x, 1) # Average over all frames within a video

        return softmax_output, net





