import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_xor.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))
# x_data = xy[0:-1]
# y_data = xy[-1]

# x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32, [None, 2]) # x1, x2
Y = tf.placeholder(tf.float32, [None, 1]) # y
# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)


W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0))
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0))
b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")
# W1 = tf.Variable(tf.random_uniform([4, 2], -1.0, 1.0))
# W2 = tf.Variable(tf.random_uniform([2, 4], -1.0, 1.0))
# b1 = tf.Variable(tf.zeros([2]), name="Bias1")
# b2 = tf.Variable(tf.zeros([1]), name="Bias2")


# L2 = tf.nn.relu(tf.matmul(X, W1) + b1)
L2 = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y)*tf.log(1-hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)

    for step in xrange(20001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data})

    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy], feed_dict={X: x_data, Y: y_data})
    # print sess.run(correct_prediction, feed_dict={X: x_data, Y: y_data})
    # print 'Accuracy:', accuracy.eval({X: x_data, Y: y_data})

    # a = sess.run([hypothesis], feed_dict={X: [[0, 0], [1, 1]]})
    # print a

