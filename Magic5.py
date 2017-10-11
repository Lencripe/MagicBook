from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys

from input_data_classification import InputData
from regression_input_data import InputDataReg


import tensorflow as tf

FLAGS = None

convLayer3 = []
def regNetwork(conv3):
  W_fc1 = weight_variable([16 * 16 * 128, 1024])
  b_fc1 = bias_variable([1024])
  h_pool3_flat = tf.reshape(conv3, [-1, 16 * 16 * 128])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.


  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 512])
  b_fc2 = bias_variable([512])
  h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)

  W_fc3 = weight_variable([512, 256])
  b_fc3 = bias_variable([256])
  h_fc3 = tf.nn.relu(tf.matmul(h_fc2, W_fc3) + b_fc3)

  w_fc4 = weight_variable([256, 4])
  b_fc4 = bias_variable([4])
  h_fc4 = tf.nn.relu(tf.matmul(h_fc3, w_fc4) + b_fc4)

  return h_fc4
def deepnn(x):
  """deepnn builds the graph for a deep net for classifying digits.
    Args:
      x: an input tensor with the dimensions (N_examples, 784), where 784 is the
      number of pixels in a standard MNIST image.
    Returns:
      A tuple (y, keep_prob). y is a tensor of shape (N_examples, 10), with values
      equal to the logits of classifying the digit into one of 10 classes (the
      digits 0-9). keep_prob is a scalar placeholder for the probability of
      dropout.
    """
  # Reshape to use within a convolutional neural net.
  # Last dimension is for "features" - there is only one here, since images are
  # grayscale -- it would be 3 for an RGB image, 4 for RGBA, etc.
  x_image = tf.reshape(x, [-1, 128, 128, 3])

  # First convolutional layer - maps one grayscale image to 32 feature maps.
  W_conv1 = weight_variable([5, 5, 3, 32])
  b_conv1 = bias_variable([32])
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

  # Pooling layer - downsamples by 2X.
  h_pool1 = max_pool_2x2(h_conv1)

  # Second convolutional layer -- maps 32 feature maps to 64.
  W_conv2 = weight_variable([5, 5, 32, 64])
  b_conv2 = bias_variable([64])
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)

  # Second pooling layer.
  h_pool2 = max_pool_2x2(h_conv2)

  W_conv3 = weight_variable([5, 5, 64, 128])
  b_conv3 = bias_variable([128])
  h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)

  # Third pooling layer.
  h_pool3 = max_pool_2x2(h_conv3)

  # Fully connected layer 1 -- after 2 round of downsampling, our 28x28 image
  # is down to 7x7x64 feature maps -- maps this to 1024 features.
  W_fc1 = weight_variable([16 * 16 * 128, 1024])
  b_fc1 = bias_variable([1024])

  h_pool3_flat = tf.reshape(h_pool3, [-1, 16 * 16 * 128])
  h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

  # Dropout - controls the complexity of the model, prevents co-adaptation of
  # features.
  keep_prob = tf.placeholder(tf.float32)

  # Map the 1024 features to 10 classes, one for each digit
  W_fc2 = weight_variable([1024, 2])
  b_fc2 = bias_variable([2])

  y_conv = tf.matmul(h_fc1, W_fc2) + b_fc2
  return y_conv, h_pool3_flat

def conv2d(x, W):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
  """max_pool_2x2 downsamples a feature map by 2X."""
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
  """weight_variable generates a weight variable of a given shape."""
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)


def bias_variable(shape):
  """bias_variable generates a bias variable of a given shape."""
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)


def main(_):
  # Import data
  ##mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)
  input_data = InputData(one_hot=True)
  input_data_reg = InputDataReg(one_hot=True)

  class_graph = tf.Graph()
  with class_graph.as_default():

    # Create the model
    x = tf.placeholder(tf.float32, [None, 49152])

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 4])

    # Build the graph for the deep net
    y_conv, conv_3 = deepnn(x)
    final_cordinates = regNetwork(conv_3)
    loss = tf.reduce_mean(tf.nn.l2_loss(final_cordinates - y_))
    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean((tf.nn.l2_loss(y_conv - y_)) )
    saver = tf.train.import_meta_graph('SavedModels/trained_model.ckpt.meta')


  with tf.Session(graph=class_graph) as sess:
    sess.run(tf.global_variables_initializer())
    sess.run(tf.assign)
    saver.restore(sess,"SavedModels/trained_model.ckpt")

    # sess.run(tf.global_variables_initializer())
    for i in range(4000):
    #   ##batch = mnist.train.next_batch(50)
      the_data = input_data_reg.test_data_random_batch(batch_size=500)
      if i % 100 == 0:
        batch = input_data.train_data_next_batch(batch_size=100)
        images = the_data[0]
        codinates = the_data[2]
        train_accuracy = accuracy.eval(feed_dict={
          x: images, y_: codinates})


    #     print('step %d, training accuracy %g' % (i, train_accuracy))

    # print('test accuracy %g' % accuracy.eval(feed_dict={
    #       # x: input_data.test.images, y_: input_data.test.labels, keep_prob: 1.0}))
    #       x: images, y_: labels, keep_prob: 1.0}))
    # train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    #   #train_step.run(feed_dict={x: images, y_: labels, keep_prob: 0.5})
    print('test accuracy %g' % accuracy.eval(feed_dict={
      # x: input_data.test.images, y_: input_data.test.labels, keep_prob: 1.0}))
      x: images, y_: codinates}))
    #   savepath = saver.save(sess,"SavedModels/trained_model.ckpt")
    #  print("The model was saved at " + savepath)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str,
                      default='/tmp/tensorflow/mnist/input_data',
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)