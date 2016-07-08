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
    return tf.nn.conv2d(x, W, strides=[1, 2, 2, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

if __name__ == '__main__':

    # Get some data
    dataset         = ud.HdfDataset('data/augmented_b.storage')
    signal_shape    = 170**2
    n_classes       = 12

    # Prepare tensorflow signal holders
    x = tf.placeholder(tf.float32, shape=[None, signal_shape])
    Y = tf.placeholder("float", [None, n_classes])

    # Start with fully connected layer to shrink the image
    W_aa = weight_variable([signal_shape, 100**2])
    b_aa = bias_variable([100**2])

    layer_aa = tf.matmul(x, W_aa)
    layer_aa = tf.add(layer_aa, b_aa)
    layer_aa = tf.nn.sigmoid(layer_aa)

    # Extract 16 features convolutionally
    x_image = tf.reshape(layer_aa, [-1, 100, 100, 1]) 
    W_conv1 = weight_variable([5, 5, 1, 16]) 
    b_conv1 = bias_variable([16])

    # Image becomes 50x50
    layer_bb = conv2d(x_image, W_conv1)
    layer_bb = tf.nn.relu(layer_bb + b_conv1)

    # Another fully connected layer
    x_flat = tf.reshape(layer_bb, [-1, 16*50*50])
    W_bb = weight_variable([16*50*50, 20*20])
    b_bb = bias_variable([20*20])

    # Image is now 20x20
    layer_cc = tf.matmul(x_flat, W_bb)
    layer_cc = tf.add(layer_cc, b_bb)
    layer_cc = tf.nn.sigmoid(layer_cc)

    # Try to extract 4 more features
    x_image_cc = tf.reshape(layer_cc, [-1, 20, 20, 1])
    W_conv2 = weight_variable([5, 5, 1, 4])
    b_conv2 = bias_variable([4])

    # Image becomes 10x10
    layer_dd = conv2d(x_image_cc, W_conv2)
    layer_dd = tf.nn.relu(layer_dd + b_conv2)

    # Final fully connected layer!
    x_flat_ee = tf.reshape(layer_dd, [-1, 4*10*10])
    W_ee = weight_variable([4*10*10, 256])
    b_ee = bias_variable([256])

    layer_ee = tf.matmul(x_flat_ee, W_ee)
    layer_ee = tf.add(layer_ee, b_ee)
    layer_ee = tf.nn.sigmoid(layer_ee)

    # Readout
    W_ff = weight_variable([256, n_classes])
    b_ff = bias_variable([n_classes])

    layer_ff = tf.matmul(layer_ee, W_ff)
    layer_ff = tf.add(layer_ff, b_ff)
    layer_ff = tf.nn.sigmoid(layer_ff)

    # What to minimize
    cost = tf.reduce_mean(tf.pow(Y - layer_ff, 2))
    optimizer = tf.train.AdamOptimizer(0.01).minimize(cost)

    learning = []
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        # Work hard
        for it in range(2000):
            si, la = dataset.next_batch(16)
            fd = { x : si, Y : la }
            _, c = sess.run([optimizer, cost], feed_dict = fd)

            learning.append(c)
            print 'cost {} at {}'.format(c, it)

        fd = { x : [si[-1]] }
        scores = sess.run(layer_ff, feed_dict = fd)

    print scores.shape
    plt.plot(scores)
    plt.savefig('tmp/nine.png')
    plt.clf()

    plt.plot(learning)
    plt.savefig('tmp/learning.png')
    plt.clf()
