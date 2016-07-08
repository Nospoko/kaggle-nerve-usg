import matplotlib as mpl
mpl.use('Agg')
import tensorflow as tf
from utils import data as ud
from datasets import basic as db
from matplotlib import pyplot as plt

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':

    # Get some data
    masked = db.masked_records()
    miniset = db.chop_image(masked[0], 20)
    signal = miniset[0][0]

    # Prepare tensorflow signal holders
    x = tf.placeholder(tf.float32, shape=[None, 170**2])
    x_image = tf.reshape(x, [-1, 170, 170, 1]) 

    # and conv filters
    W_conv1 = weight_variable([5, 5, 1, 9]) 

    # Convolve
    first = conv2d(x_image, W_conv1)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        fd = { x : [signal] }
        scores = sess.run(first, feed_dict = fd)

    # Reshape scores into a list of images
    final = []
    for it in range(9):
        final.append(scores[0, :, :, it])

    ud.show_nine(final)
    plt.savefig('tmp/nine.png')
    plt.clf()
