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

def main():
    # Dataset path
    train_list = 'video_train_list_middle_cam0.txt' # train: cam0 + cam1 (defined in dataset.py), test: cam1, val: cam0
    test_list = 'video_test_list_middle_cam1.txt'
    val_list = 'video_val_list_middle_cam0.txt'

    # Learning params
    # learning_rate = 0.001



    training_iters = 5000  # 10 epochs
    batch_size = 8
    display_step = 100
    test_step = 200  # 0.5 epoch



    n_hidden = 64

    lstm_hidden = 64
    sequence = 20

    im_height = 112
    im_width = 112

    # Network params
    n_classes = 10
    keep_rate = 0.5

    learning_rate = 0.001

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, sequence, im_height, im_width, 3])
    target_x = tf.placeholder(tf.float32, [batch_size, sequence, im_height, im_width, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])

    keep_var = tf.placeholder(tf.float32)

    input_imgs = tf.reshape(x, [-1, im_height, im_width, 3])
    input_imgs = input_imgs * 255

    tf.summary.image("input frames", input_imgs, 20)

    # Model
    # pred, encoded = Model.alexnet_vae(x, keep_var, n_class=n_classes, hidden_size=n_hidden * 2)

    # encoder_input = tf.reshape(x, [-1, 112, 112, 3])
    with tf.variable_scope("model") as scope:
        pred, encoded = Model.alexnet_encoder(x, keep_var, n_classes, n_hidden * 2)
    with tf.variable_scope("model_cross") as scope:
        pred_cross, encoded_cross = Model.alexnet_encoder(target_x, keep_var, n_classes, n_hidden * 2)

    # mean = encoded[:, :n_hidden]
    # stddev = tf.sqrt(tf.exp(encoded[:, n_hidden:]))
    # epsilon = tf.random_normal([tf.shape(mean)[0], n_hidden])
    # input_sample = mean + epsilon * stddev
    # input_sample = mean
    output_tensor, generated_imgs = Model.decoder(encoded[:, :n_hidden])
    output_imgs = tf.reshape(output_tensor, [-1, im_height, im_width, 3])

    output_imgs = output_imgs * 255
    tf.summary.image("generated frames", output_imgs, 20)


    with tf.variable_scope("model_cross", reuse=True) as scope:
        pred_cross_gen, encoded_cross_gen = Model.alexnet_encoder(generated_imgs, keep_var, n_classes, n_hidden * 2)






    x_flattened = tf.reshape(target_x, [-1, 112 * 112 * 3])
    rec_loss = Model.get_reconstruction_cost(x_flattened, output_tensor)
    rec_loss_feature = Model.get_reconstruction_cost_feature(encoded_cross_gen, encoded_cross)
    tf.summary.scalar('rec_loss_loss', rec_loss)
    tf.summary.scalar('rec_loss_feature_loss', rec_loss_feature)




        # Loss and optimizer
    cross_entropy_loss_1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_cross_gen, y))
    cross_entropy_loss_2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred_cross, y))

    tf.summary.scalar('cross_entropy_loss_1', cross_entropy_loss_1)
    tf.summary.scalar('cross_entropy_loss_2', cross_entropy_loss_2)

    # tf.summary.scalar('perception_loss_loss', Perception_loss)
    # tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

    loss = 10 * cross_entropy_loss_1 + 10 * cross_entropy_loss_2 + rec_loss + rec_loss_feature
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)

    optimizer = layers.optimize_loss(loss, tf.contrib.framework.get_or_create_global_step(
    ), learning_rate=learning_rate, optimizer='Adam', update_ops=[])


    # Evaluation
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    merged = tf.summary.merge_all()

    # Init
    init = tf.initialize_all_variables()

    # Load dataset
    dataset = Dataset(train_list, test_list, val_list, n_classes=n_classes, shuffleType='seq', seqLength=sequence,
                      CNN_type='Alex')
    saver_all = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        print 'Loading weights'
        sess.run(init)
        with tf.variable_scope("model") as scope:
            load_with_skip('bvlc_alexnet.npy', sess, ['fc8'])  # Skip weights from fc8
        with tf.variable_scope("model_cross") as scope:
            load_with_skip('bvlc_alexnet.npy', sess, ['fc8'])
        # print 'Loading weights'
        # saver_all.restore(sess, 'save_model_cross/finetuned_cross_loss.ckpt')

        # print 'Extracting features'

        val_feature = np.zeros((batch_size, n_hidden * 2 * sequence), int)
        val_label = np.zeros((batch_size, 1), int)
        for _ in range(int(dataset.val_size / batch_size)):
            batch_tx, batch_ty, label_ty, batch_target = dataset.next_batch_cross(batch_size, 'train')
            output_feature = sess.run(encoded, feed_dict={x: batch_tx, target_x: batch_target, y: batch_ty, keep_var: 1.})
            output_feature = output_feature.reshape(batch_size, sequence, n_hidden * 2)
            output_feature = output_feature.reshape(batch_size, sequence * n_hidden * 2)
            val_feature = np.concatenate((val_feature, output_feature))
            val_label = np.concatenate((val_label, label_ty))

        val_feature = val_feature[8:, :]
        val_label = val_label[8:, :]

        test_feature = np.zeros((batch_size, n_hidden * 2 * sequence), int)
        test_label = np.zeros((batch_size, 1), int)
        for _ in range(int(dataset.test_size / batch_size)):
            batch_tx, batch_ty, label_ty, batch_target = dataset.next_batch_cross(batch_size, 'test')
            output_feature = sess.run(encoded,
                                      feed_dict={x: batch_tx, target_x: batch_target, y: batch_ty, keep_var: 1.})
            output_feature = output_feature.reshape(batch_size, sequence, n_hidden * 2)
            output_feature = output_feature.reshape(batch_size, sequence * n_hidden * 2)
            test_feature = np.concatenate((test_feature, output_feature))
            test_label = np.concatenate((test_label, label_ty))

        test_feature = test_feature[8:, :]
        test_label = test_label[8:, :]

        clf = svm.SVC()
        val_label = np.ravel(val_label)
        test_label = np.ravel(test_label)

        clf.fit(val_feature, val_label)

        SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
            max_iter=-1, probability=False, random_state=None, shrinking=True,
            tol=0.001, verbose=False)

        predicted = clf.predict(test_feature)

        print accuracy_score(test_label, predicted)


if __name__ == '__main__':
    main()