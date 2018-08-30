"""
implementation of Endnet: Sparse AutoEncoder Network for Endmember Extraction and Hyperspectral Unmixing
2018, TGRS, Savas Ozkan.
"""
from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat
from numpy import linalg as LA
import matplotlib.pyplot as plt

# Import data
data = loadmat('data/samson/samson.mat')
data = data['V'].transpose()
gt = loadmat('data/samson/end3.mat')

# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("data/", one_hot=True)

# Training Parameters
num_example = data.shape[0]
learning_rate = 0.01
training_epochs = 100
batch_size = 256

# display_step = 10
examples_to_show = 10

# Network Parameters
num_hidden_1 = 64 # 1st layer num features
num_hidden_2 = 3 # 2nd layer num features (the latent dim)
num_input = 156 # MNIST data input (img shape: 28*28)

# tf Graph input (only pictures)
X = tf.placeholder("float", [None, num_input])

weights = {
    'encoder_h1': tf.Variable(tf.random_normal([num_input, num_hidden_1])),
    'encoder_h2': tf.Variable(tf.random_normal([num_hidden_1, num_hidden_2])),
    'decoder': tf.Variable(tf.random_normal([num_hidden_2, num_input])),
}
biases = {
    'encoder_b1': tf.Variable(tf.random_normal([num_hidden_1])),
    'encoder_b2': tf.Variable(tf.random_normal([num_hidden_2])),
    'decoder': tf.Variable(tf.random_normal([num_input])),
}


# Building the encoder
def encoder(x):
    # Encoder Hidden layer with sigmoid activation #1
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    # Encoder Hidden layer with sigmoid activation #2
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2


# Building the decoder
def decoder(x):
    # Decoder Hidden layer with sigmoid activation #1
    layer = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder']),
                                 biases['decoder']))
    return layer


# Construct model
encoder_op = encoder(X)
decoder_op = decoder(encoder_op)

# Prediction
y_pred = decoder_op
# Targets (Labels) are the input data.
y_true = X

# Define loss and optimizer, minimize the squared error
loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
# optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start Training
# Start a new TF session
saver = tf.train.Saver()
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Training
    for epoch in range(training_epochs):
        epoch_loss = 0
        for i in range(num_example//batch_size + 1):
            # Prepare Data
            if i < num_example//batch_size:
                batch_x = data[i * batch_size:(i+1) * batch_size, :]
            else:
                batch_x = data[i * batch_size: , :]

            # Run optimization op (backprop) and cost op (to get loss value)
            _, l = sess.run([optimizer, loss], feed_dict={X: batch_x})
            epoch_loss += l
            # Display logs per step
            # if i % display_step == 0:
            #     print('Step %i: Minibatch Loss: %f' % (i, l))
        # print("epoch %i: loss: %f" % (epoch, epoch_loss/(i+1)))

    saver.save(sess, 'model/my_test_model.ckpt')
    print("optimization finished.")

    # # Testing
    params = tf.trainable_variables()
    # print("Trainable variables:------------------------")
    # # 循环列出参数
    # for idx, v in enumerate(params):
    #     print("  param {:3}: {:15}   {}".format(idx, str(v.get_shape()), v.name))
    endmember = sess.run(params[2])
    gt_endmember = gt['M'].transpose()

    fig, axes = plt.subplots(1, 3)
    for i in range(3):
        axes[i].plot(range(156), endmember[i,:], range(156), gt_endmember[i,:])
    fig

    # rmse = LA.norm(endmember - gt_endmember)
    # print(rmse)