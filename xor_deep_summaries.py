import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_xor.txt', unpack=True)
x_data = np.transpose(xy[0:-1])
y_data = np.reshape(xy[-1], (4, 1))
# x_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
# y_data = np.array([[0], [1], [1], [0]])

X = tf.placeholder(tf.float32, shape=[None, 2], name='X-input')
Y = tf.placeholder(tf.float32, shape=[None, 1], name='Y-input')


W1 = tf.Variable(tf.random_uniform([2, 2], -1.0, 1.0), name='Weight1')
W2 = tf.Variable(tf.random_uniform([2, 1], -1.0, 1.0), name='Weight2')

b1 = tf.Variable(tf.zeros([2]), name="Bias1")
b2 = tf.Variable(tf.zeros([1]), name="Bias2")

with tf.name_scope("layer2") as scope:
    L2 = tf.sigmoid(tf.matmul(X, W1) + b1)

with tf.name_scope("layer3") as scope:
    hypothesis = tf.sigmoid(tf.matmul(L2, W2) + b2)

with tf.name_scope("cost") as scope:
    cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))
    cost_sum = tf.scalar_summary("cost", cost)

with tf.name_scope("cost") as scope:
    optimizer = tf.train.GradientDescentOptimizer(0.1)
    train = optimizer.minimize(cost)

with tf.name_scope("accuracy") as scope:
    # Test model
    correct_prediction = tf.equal(tf.floor(hypothesis + 0.5), Y)
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    acc = tf.scalar_summary("accuracy", accuracy)


w1_hist = tf.histogram_summary("weights1", W1)
w2_hist = tf.histogram_summary("weights2", W1)

b1_hist = tf.histogram_summary("biases1", b1)
b2_hist = tf.histogram_summary("biases2", b2)

y_hist = tf.histogram_summary("y", Y)

init = tf.initialize_all_variables()


with tf.Session() as sess:
    sess.run(init)

    #tensorboard --logdir=/log
    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("/log/xor_logs", sess.graph_def)

    for step in xrange(10001):
        sess.run(train, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            summary, c = sess.run([merged, cost], feed_dict={X: x_data, Y: y_data})
            writer.add_summary(summary, step)
            print step, c

    print sess.run([hypothesis, tf.floor(hypothesis + 0.5), correct_prediction, accuracy],
                   feed_dict={X: x_data, Y: y_data})

