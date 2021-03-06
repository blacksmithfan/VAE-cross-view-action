import tensorflow as tf
import sys
from model import Model
from dataset import Dataset
from network import *
from datetime import datetime
from tensorflow.contrib import layers
import numpy as NP
from scipy.misc import imsave
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

def main():
    # Dataset path
    train_list = 'video_train_list_middle_cam0.txt'
    test_list = 'video_test_list_middle_cam0.txt'
    val_list = 'video_val_list_middle_cam0.txt'

    # Learning params
    learning_rate = 0.001
    training_iters = 30000  # 10 epochs
    batch_size = 8
    display_step = 30
    test_step = 10  # 0.5 epoch

    n_hidden = 64

    lstm_hidden = 64
    sequence = 20

    im_height = 112
    im_width = 112

    # Network params
    n_classes = 10
    keep_rate = 0.5

    clf = svm.SVC()

    # Graph input
    x = tf.placeholder(tf.float32, [batch_size, sequence, im_height, im_width, 3])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_var = tf.placeholder(tf.float32)

    input_imgs = tf.reshape(x, [-1, im_height, im_width, 3])
    input_imgs = input_imgs * 255

    tf.summary.image("input frames", input_imgs, 20)

    # Model
    # pred, encoded = Model.alexnet_vae(x, keep_var, n_class=n_classes, hidden_size=n_hidden * 2)

    # encoder_input = tf.reshape(x, [-1, 112, 112, 3])
    pred, encoded = Model.encoder(x, n_hidden * 2, n_classes, keep_var)

    mean = encoded[:, :n_hidden]
    # stddev = tf.sqrt(tf.exp(encoded[:, n_hidden:]))
    # epsilon = tf.random_normal([tf.shape(mean)[0], n_hidden])
    # input_sample = mean + epsilon * stddev
    input_sample = mean
    output_tensor, generated_imgs = Model.decoder(input_sample)
    # Obtain output images
    output_imgs = tf.reshape(output_tensor, [-1, im_height, im_width, 3])

    output_imgs = output_imgs * 255
    tf.summary.image("generated frames", output_imgs, 20)




    # input_img_vector = layers.flatten(x_gray)
    x_flattened = tf.reshape(x, [-1, 112 * 112 * 3])
    rec_loss = Model.get_reconstruction_cost(x_flattened, output_tensor)
    tf.summary.scalar('rec_loss_loss', rec_loss)

    # Perception loss
    input_x = tf.reshape(x, [-1, 112, 112, 3])
    input_x = tf.image.resize_images(input_x, [227, 227])
    input_x = input_x * 255

    with tf.variable_scope("model") as scope:
        input_tensor_features= Model.alexnet_lstm(input_x, keep_var, sequence, lstm_hidden, batch_size)
        input_tensor_features = tf.reshape(input_tensor_features, [-1, batch_size * lstm_hidden])

        generated_imgs = tf.image.resize_images(generated_imgs, [227, 227])
        generated_imgs = generated_imgs * 255

    with tf.variable_scope("model", reuse=True) as scope:
        output_tensor_features = Model.alexnet_lstm(input_x, keep_var, sequence, lstm_hidden, batch_size)
        output_tensor_features = tf.reshape(output_tensor_features, [-1, batch_size * lstm_hidden])

        Perception_loss = Model.get_reconstruction_cost(input_tensor_features, output_tensor_features)


        # Loss and optimizer
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))

    tf.summary.scalar('perception_loss_loss', Perception_loss)
    tf.summary.scalar('cross_entropy_loss', cross_entropy_loss)

    loss = rec_loss + cross_entropy_loss + Perception_loss
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
    dataset = Dataset(train_list, test_list, val_list, '0', n_classes=n_classes, shuffleType='seq', seqLength=sequence,
                      CNN_type='Alex')
    saver_all = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        print 'Init variable'
        sess.run(init)
        # print 'Loading weights'
        # saver_all.restore(sess, 'save_model/finetuned_vae_save.ckpt')

        train_writer = tf.summary.FileWriter('./logs/train', sess.graph)
        test_writer = tf.summary.FileWriter('./logs/test', sess.graph)
        # Load pretrained model
        with tf.variable_scope("model") as scope:
            load_with_skip('bvlc_alexnet.npy', sess, ['fc8', 'fc7', 'fc6'])  # Skip weights from fc8

            print 'Start training'
            step = 1
            epoch = 1
            while step < training_iters:
                batch_xs, batch_ys, _ = dataset.next_batch(batch_size, 'train')
                output_merged, _ = sess.run([merged, optimizer], feed_dict={x: batch_xs, y: batch_ys, keep_var: keep_rate})

                train_writer.add_summary(output_merged, step)

                # Display testing status
                if step % test_step == 0:
                    test_acc = 0.
                    test_count = 0
                    test_feature = np.empty((batch_size, n_hidden * 2 * sequence), int)
                    test_label = np.empty((batch_size, 1), int)
                    for _ in range(int(dataset.test_size / batch_size)):
                        batch_tx, batch_ty, label_ty = dataset.next_batch(batch_size, 'test')
                        acc, output_feature = sess.run([accuracy, encoded], feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                        output_feature = output_feature.reshape(batch_size, sequence, n_hidden * 2)
                        output_feature = output_feature.reshape(batch_size, sequence * n_hidden * 2)
                        test_feature = np.concatenate((test_feature, output_feature))
                        test_label = np.concatenate((test_label, label_ty))
                        test_acc += acc
                        test_count += 1
                    test_acc /= test_count
                    print >> sys.stderr, "{} Iter {}: Testing Accuracy = {:.4f}".format(datetime.now(), step, test_acc)



                    train_feature = np.empty((batch_size, n_hidden * 2 * sequence), int)
                    train_label = np.empty((batch_size, 1), int)
                    for _ in range(int(dataset.train_size / batch_size)):
                        batch_tx, batch_ty, label_ty = dataset.next_batch(batch_size, 'train')
                        acc, output_feature = sess.run([accuracy, encoded], feed_dict={x: batch_tx, y: batch_ty, keep_var: 1.})
                        output_feature = output_feature.reshape(batch_size, sequence, n_hidden * 2)
                        output_feature = output_feature.reshape(batch_size, sequence * n_hidden * 2)
                        train_feature = np.concatenate((train_feature, output_feature))
                        train_label = np.concatenate((train_label, label_ty))


                    clf.fit(train_feature, train_label)

                    SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0, decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
                        max_iter=-1, probability=False, random_state=None, shrinking=True,
                        tol=0.001, verbose=False)

                    predicted = clf.predict(test_feature)

                    print predicted
                    print accuracy_score(test_label, predicted)

                    sample_video, sample_label = Dataset.test_video('sample_train_video.txt', batch_size, n_classes)
                    output_merged_test = sess.run(merged, feed_dict={x: sample_video, y: sample_label, keep_var: 1.})

                    test_writer.add_summary(output_merged_test, step)

                # Display training status
                if step % display_step == 0:
                    acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                    batch_loss = sess.run(loss, feed_dict={x: batch_xs, y: batch_ys, keep_var: 1.})
                    print >> sys.stderr, "Iter {}, Epoch {}: Training Loss = " \
                                         "{:.4f}, Accuracy = {:.4f}".format(step, epoch, batch_loss, acc)
                    epoch += 1

                step += 1
            print "Finish!"
            save_path = saver_all.save(sess, "save_model2/finetuned_vae_save_perception_loss.ckpt")
            print("Model saved in file: %s" % save_path)


if __name__ == '__main__':
    main()