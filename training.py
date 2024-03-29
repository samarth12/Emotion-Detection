#! /usr/bin/env python
import json
import logging
import os
import time

import tensorflow as tf
import numpy as np
import datetime
import preprocessing
from cnn import TextCNN
from tensorflow.contrib import learn
from sklearn.cross_validation import train_test_split

logging.getLogger().setLevel(logging.INFO)

def train():
    train_file = 'data/emotion Phrases.csv'
    x_raw, y_raw, df, labels, embedding_mat = preprocessing.load_data(train_file)

    parameter_file = './parameters.json'
    params = json.loads(open(parameter_file).read())

    max_document_length = max([len(x.split(' ')) for x in x_raw])
    logging.info('The maximum length of all sentences: {}'.format(max_document_length))
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_raw)))
    y = np.array(y_raw)

    # print x.shape

    x_, x_test, y_, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    shuffle_indices = np.random.permutation(np.arange(len(y_)))
    x_shuffled = x_[shuffle_indices]
    y_shuffled = y_[shuffle_indices]
    x_train, x_dev, y_train, y_dev = train_test_split(x_shuffled, y_shuffled, test_size=0.2)

#    with open('labels.json', 'w') as outfile:
#        json.dump(labels, outfile, indent=4)

    logging.info('x_train: {}, x_dev: {}, x_test: {}'.format(len(x_train), len(x_dev), len(x_test)))
    logging.info('y_train: {}, y_dev: {}, y_test: {}'.format(len(y_train), len(y_dev), len(y_test)))

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=9017,
                embedding_size=params['embedding_dim'],
                filter_sizes=list(map(int, params['filter_sizes'].split(","))),
                num_filters=params['num_filters'], embedding_mat=embedding_mat,
                l2_reg_lambda=params['l2_reg_lambda'])

            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "trained_model_" + timestamp))

            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables())

            # One training step: train the model with one batch
            def train_step(x_batch, y_batch):
                feed_dict = {
                    cnn.input_x: x_batch,
                    cnn.input_y: y_batch,
                    cnn.dropout_keep_prob: params['dropout_keep_prob']}
                _, step, loss, acc = sess.run([train_op, global_step, cnn.loss, cnn.accuracy], feed_dict)

            # One evaluation step: evaluate the model with one batch
            def dev_step(x_batch, y_batch):
                feed_dict = {cnn.input_x: x_batch, cnn.input_y: y_batch,
                             cnn.dropout_keep_prob: 1.0}
                step, loss, acc, num_correct = sess.run([global_step, cnn.loss, cnn.accuracy, cnn.num_correct],
                                                        feed_dict)
                return num_correct

            # Save the word_to_id map since predict.py needs it
            vocab_processor.save(os.path.join(out_dir, "vocab.pickle"))
            sess.run(tf.global_variables_initializer())

            print "Loading Embeddings !"
            initW = preprocessing.load_embedding_vectors(vocab_processor.vocabulary_)
            #
            #x = np.shape(a)
            #print(x)
            print(np.shape(initW))
            sess.run(cnn.W.assign(initW))

            print "Loaded Embeddings !"

            # Training starts here
            train_batches = preprocessing.generate_batches(list(zip(x_train, y_train)), params['batch_size'],
                                                   params['num_epochs'])
            best_accuracy, best_at_step = 0, 0


            for train_batch in train_batches:
                if len(train_batch) == 0:
                    continue
                x_train_batch, y_train_batch = zip(*train_batch)
                train_step(x_train_batch, y_train_batch)
                current_step = tf.train.global_step(sess, global_step)

                if current_step % params['evaluate_every'] == 0:
                    dev_batches = preprocessing.generate_batches(list(zip(x_dev, y_dev)), params['batch_size'], 1)
                    total_dev_correct = 0
                    for dev_batch in dev_batches:
                        if len(dev_batch) == 0:
                            continue
                        x_dev_batch, y_dev_batch = zip(*dev_batch)
                        num_dev_correct = dev_step(x_dev_batch, y_dev_batch)
                        total_dev_correct += num_dev_correct

                    dev_accuracy = float(total_dev_correct) / len(y_dev)
                    logging.critical('Accuracy on dev set: {}'.format(dev_accuracy))

                    if dev_accuracy >= best_accuracy:
                        best_accuracy, best_at_step = dev_accuracy, current_step
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        logging.critical('Saved model {} at step {}'.format(path, best_at_step))
                        logging.critical('Best accuracy {} at step {}'.format(best_accuracy, best_at_step))

            test_batches = preprocessing.generate_batches(list(zip(x_test, y_test)), params['batch_size'], 1)
            total_test_correct = 0
            for test_batch in test_batches:
                if len(test_batch) == 0:
                    continue
                print "Non Zero Length"
                x_test_batch, y_test_batch = zip(*test_batch)
                num_test_correct = dev_step(x_test_batch, y_test_batch)
                total_test_correct += num_test_correct

            test_accuracy = float(total_test_correct) / len(y_test)

            train_batches = preprocessing.generate_batches(list(zip(x_train, y_train)), params['batch_size'], 1)

            total_train_correct = 0
            for train_batch in train_batches:
                if len(train_batch) == 0:
                    continue
                print "Non Zero Length"
                x_train_batch, y_train_batch = zip(*train_batch)
                num_test_correct = dev_step(x_train_batch, y_train_batch)
                total_train_correct += num_test_correct

            train_accuracy = float(total_train_correct) / len(y_train)

        print 'Accuracy on test set is {} based on the best model'.format(test_accuracy)
        print 'Accuracy on train set is {} based on the best model'.format(train_accuracy)
        # logging.critical('Accuracy on test set is {} based on the best model {}'.format(test_accuracy, path))
        logging.critical('The training is complete')


if __name__ == '__main__':
    # python3 train_cnn.py
    train()
