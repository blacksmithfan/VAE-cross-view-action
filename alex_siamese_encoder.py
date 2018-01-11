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
import inference

def main():
    # Dataset path
    train_list = 'video_train_list_middle_cam0.txt' # train: cam0 + cam1 (defined in dataset.py), test: cam1, val: cam0
    test_list = 'video_test_list_middle_cam1.txt'
    val_list = 'video_val_list_middle_cam0.txt'

    # Learning params

    training_iters = 10000  # 10 epochs
    batch_size = 16
    display_step = 10
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
    keep_var = tf.placeholder(tf.float32)



    input_imgs = tf.reshape(x, [-1, im_height, im_width, 3])
    input_imgs = input_imgs * 255

    tf.summary.image("input frames", input_imgs, 20)

    with tf.variable_scope("model") as scope:
        pred, encoded = Model.alexnet_encoder(x, keep_var, n_classes, n_hidden * 2)

    output_tensor, generated_imgs = Model.decoder(encoded[:, :n_hidden])
    output_imgs = tf.reshape(output_tensor, [-1, im_height, im_width, 3])

    output_imgs = output_imgs * 255
    tf.summary.image("generated frames", output_imgs, 20)


    siamese = inference.siamese(keep_var, output_tensor, target_x, y)


    # cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_cross, tf.concat(0, [y, y])))

    # x_flattened = tf.reshape(target_x, [-1, 112 * 112 * 3])
    # rec_loss = Model.get_reconstruction_cost(x_flattened, output_tensor)
    # rec_loss_feature = Model.get_reconstruction_cost_feature(encoded_cross_gen, encoded_cross)
    # tf.summary.scalar('rec_loss_loss', rec_loss)
    # # tf.summary.scalar('rec_loss_feature_loss', rec_loss_feature)
    #
    # tf.summary.scalar('cross_entropy_loss_1', cross_entropy_loss)


    # loss = 10 * cross_entropy_loss + rec_loss

    x_flattened = tf.reshape(x_cross, [-1, 112 * 112 * 3])
    rec_loss = Model.get_reconstruction_cost(x_flattened, output_tensor)

    loss = siamese.loss + rec_loss

    optimizer = layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(
    ), learning_rate=learning_rate, optimizer='Adam', update_ops=[])


    # Evaluation
    # correct_pred = tf.equal(tf.argmax(pred_cross, 1), tf.argmax(tf.concat(0, [y, y]), 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    merged = tf.summary.merge_all()

    # Init
    init = tf.initialize_all_variables()

    # Load dataset
    dataset_source = Dataset(train_list, test_list, val_list, '1', n_classes=n_classes, shuffleType='seq', seqLength=sequence,
                      CNN_type='Alex')

    dataset_target = Dataset(train_list, test_list, val_list, '1', n_classes=n_classes, shuffleType='seq', seqLength=sequence,
                      CNN_type='Alex')

    saver_all = tf.train.Saver()


    # Launch the graph
    with tf.Session() as sess:
        print 'Loading weights'
        sess.run(init)
        with tf.variable_scope("model") as scope:
            load_with_skip('bvlc_alexnet.npy', sess, ['fc8'])  # Skip weights from fc8
        with tf.variable_scope("siamese") as scope:
            load_with_skip('bvlc_alexnet.npy', sess, ['fc8'])

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

            sess.run(optimizer, feed_dict={x: batch_xs, x_cross: batch_xs_cross, target_x: batch_xs_target, y: pair_label, keep_var: keep_rate})

            if step % display_step == 0:
                batch_xs, batch_ys, batch_label, batch_xs_cross = dataset_source.next_batch_cross(batch_size, 'train')
                batch_xs_target, batch_ys_target, batch_label_target = dataset_target.next_batch(batch_size, 'train')

                pair_label = np.zeros((batch_size, 1), int)
                for i in range(batch_size):
                    if batch_label[i] == batch_label_target[i]:
                        pair_label[i] = 1

                loss_value, rec_loss_value = sess.run([siamese.loss, rec_loss], feed_dict={x: batch_xs, x_cross: batch_xs_cross, target_x: batch_xs_target, y: pair_label, keep_var: 1.})

                print >> sys.stderr, "Iter {}, Epoch {}: Training Loss = " \
                                     "{:.4f}, Rec loss = {:.4f}".format(step, epoch, loss_value, rec_loss_value)
                epoch += 1

            step += 1
        print "Finish!"
        # save_path = saver_all.save(sess, "save_model_cross_test/finetuned_cross_loss.ckpt")
        # print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main()