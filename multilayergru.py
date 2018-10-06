""" Recurrent Neural Network.
A Recurrent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)




def RNN(x, weights, biases):

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, timesteps, n_input)
    # Required shape: 'timesteps' tensors list of shape (batch_size, n_input)

    # Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    x = tf.unstack(x, timesteps, 1)
    # rnn_cell1 = tf.contrib.rnn.BasicLSTMCell(num_hidden)
    # run_cell2 = tf.contrib.rnn.BasicLSTMCell(num_hidden)
    # stack_rnn = [rnn_cell1]
    # for i in range(1, 3):
    #     stack_rnn.append(run_cell2)
    # cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)
    #


    # Define a lstm cell with tensorflow
    gru_cell1 = tf.contrib.rnn.GRUCell(num_hidden)
    gru_cell2= tf.contrib.rnn.GRUCell(num_hidden)

    stack_rnn = [gru_cell1]
    # stack_rnn.append(gru_cell2)
    #
    # stack_rnn.append(gru_cell2)
    #
    # stack_rnn.append(gru_cell2)
    #
    # stack_rnn.append(gru_cell2)
    #
    # stack_rnn.append(gru_cell2)

    # cell = tf.contrib.rnn.MultiRNNCell(
    #     [lstm_cell() for _ in range(3)])





    for i in range(1,3):
        stack_rnn.append(gru_cell2)
    #
    cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple=True)

    # Get lstm cell output



    outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']




'''
To classify images using a recurrent neural network, we consider every image
row as a sequence of pixels. Because MNIST image shape is 28*28px, we will then
handle 28 sequences of 28 steps for every sample.
'''

# Training Parameters
learning_rate = 0.001
training_steps = 2000
batch_size = 128
display_step = 200

# Network Parameters
# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
timesteps = 1 # timesteps
num_hidden = 512 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])




# Define weights

weights = {
'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}


biases = {
'out': tf.Variable(tf.random_normal([num_classes]))
}




def lstm_cell():
  return tf.contrib.rnn.GRUCell(num_hidden)




logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)


# with tf.name_scope('cross_entropy'):
#     # The raw formulation of cross-entropy,
#     #
#     # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
#     #                               reduction_indices=[1]))
#     #
#     # can be numerically unstable.
#     #
#     # So here we use tf.nn.softmax_cross_entropy_with_logits on the
#     # raw outputs of the nn_layer above, and then average across
#     # the batch.
loss_op = tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=logits)

cross_entropy = tf.reduce_mean(loss_op)


train_op = tf.train.AdamOptimizer(0.001).minimize(
    cross_entropy)


correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))

accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


# # # Define loss and optimizer
# # loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
# #     logits=logits, labels=Y))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# train_op = optimizer.minimize(loss_op)
#
# # Evaluate model (with test logits, for dropout to be disabled)
# correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#


# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()
import os
ckpt_dir = "./NEW_chk"
if not os.path.exists(ckpt_dir):
    os.makedirs(ckpt_dir)

global_step = tf.Variable(0, name='global_step', trainable=False)

# Call this after declaring all tf.Variables.
saver = tf.train.Saver()

# This variable won't be stored, since it is declared after tf.train.Saver()
non_storable_variable = tf.Variable(777)
# Start training
with tf.Session() as sess:

    # Run the initializer
    tf.global_variables_initializer().run()

    ckpt = tf.train.get_checkpoint_state(ckpt_dir)
    if ckpt and ckpt.model_checkpoint_path:
        print(ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path) # restore all variables

    start = global_step.eval() # get last global_step
    print("Start from:", start)


    # sess.run(init)

    for step in range(start, training_steps+1):
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        tf.summary.image('input', batch_x, 10)
        # ckpt = tf.train.get_checkpoint_state(ckpt_dir)
        # if ckpt and ckpt.model_checkpoint_path:
        #     # print(ckpt.model_checkpoint_path)
        #     saver.restore(sess, ckpt.model_checkpoint_path)  # restore all variables

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            acc, ff = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            # print(ff)
            # print(acc)
            global_step.assign(step).eval()
            saver.save(sess, ckpt_dir + "/model.ckpt", global_step=global_step)

            # print("Step " + str(step) )

    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))