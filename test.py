import tensorflow as tf
import sys
from model import Model
from dataset import Dataset
from network import *
from datetime import datetime
from tensorflow.contrib import layers
from scipy.misc import imsave
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import inference_slim


def main():
    # Dataset path
    train_list = 'video_train_list_middle_cam0.txt'  # train: cam0 + cam1 (defined in dataset.py), test: cam1, val: cam0
    test_list = 'video_test_list_middle_cam1.txt'
    val_list = 'video_val_list_middle_cam0.txt'

    # Learning params

    training_iters = 100  # 10 epochs
    batch_size = 120
    display_step = 1
    n_hidden = 64
    sequence = 20
    im_height = 112
    im_width = 112
    n_classes = 10
    keep_rate = 0.5
    learning_rate = 0.001

    x = tf.placeholder(tf.float32, [batch_size, sequence, im_height, im_width, 3])
    x_cross = tf.placeholder(tf.float32, [batch_size, sequence, im_height, im_width, 3])
    target_x = tf.placeholder(tf.float32, [batch_size, sequence, im_height, im_width, 3])
    y = tf.placeholder(tf.float32, [None, 1])
    y_source = tf.placeholder(tf.float32, [None, 10])
    y_target = tf.placeholder(tf.float32, [None, 10])
    keep_var = tf.placeholder(tf.float32)

    input_imgs = tf.reshape(x, [-1, im_height, im_width, 3])
    input_imgs = input_imgs * 255

    tf.summary.image("input frames", input_imgs, 20)

    pred, encoded = Model.encoder(x, n_hidden * 2, n_classes, keep_var)

    output_tensor, _ = Model.decoder(encoded[:, :n_hidden])
    output_imgs = tf.reshape(output_tensor, [-1, im_height, im_width, 3])

    output_imgs = output_imgs * 255
    tf.summary.image("generated frames", output_imgs, 20)


    x_flattened = tf.reshape(x_cross, [-1, 112 * 112 * 3])
    rec_loss = Model.get_reconstruction_cost(x_flattened, output_tensor)

    tf.summary.scalar('rec_loss_loss', rec_loss)

    # Cross-entropy loss

    loss = rec_loss


    optimizer = layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(
    ), learning_rate=learning_rate, optimizer='Adam', update_ops=[])

    merged = tf.summary.merge_all()

    # Init
    init = tf.initialize_all_variables()

    # Load dataset
    dataset_source = Dataset(train_list, test_list, val_list, '1', n_classes=n_classes, shuffleType='seq',
                             seqLength=sequence,
                             CNN_type='Alex')  # Paths output camera 0, so here '1' indicates the target view data

    dataset_target = Dataset(train_list, test_list, val_list, '0', n_classes=n_classes, shuffleType='seq',
                             seqLength=sequence,
                             CNN_type='Alex')

    saver_all = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        print 'Loading weights'
        sess.run(init)

        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)

        print 'Start training'
        step = 1
        epoch = 1
        while step < training_iters:

            batch_xs, batch_ys, batch_label, batch_xs_cross = dataset_source.next_batch_cross(batch_size, 'train')
            batch_xs_target, batch_ys_target, batch_label_target = dataset_target.next_batch(batch_size, 'train')

            pair_label = np.zeros((batch_size, 1), int)
            for i in range(batch_size):
                if batch_label[i] == batch_label_target[i]:
                    pair_label[i] = 1

            output_merged, _ = sess.run([merged, optimizer],
                                        feed_dict={x: batch_xs, x_cross: batch_xs_cross, target_x: batch_xs_target,
                                                   y: pair_label, keep_var: keep_rate,
                                                   y_source: batch_ys, y_target: batch_ys_target})
            train_writer.add_summary(output_merged, step)

            if step % display_step == 0:
                batch_xs, batch_ys, batch_label, batch_xs_cross = dataset_source.next_batch_cross(batch_size, 'train')
                batch_xs_target, batch_ys_target, batch_label_target = dataset_target.next_batch(batch_size, 'train')

                pair_label = np.zeros((batch_size, 1), int)
                for i in range(batch_size):
                    if batch_label[i] == batch_label_target[i]:
                        pair_label[i] = 1

                rec_loss_value = sess.run(rec_loss, feed_dict={x: batch_xs,
                                                                                           x_cross: batch_xs_cross,
                                                                                           target_x: batch_xs_target,
                                                                                           y_source: batch_ys,
                                                                                           y_target: batch_ys_target,
                                                                                           y: pair_label, keep_var: 1.})

                epoch += 1


            step += 1
        print "Finish!"
        save_path = saver_all.save(sess, "save_model_siamese/finetuned_cross_loss.ckpt")
        # print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main()