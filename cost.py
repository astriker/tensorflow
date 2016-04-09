import tensorflow as tf

x_data = [1., 2., 3.]
y_data = [1., 2., 3.]

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1], -10.0, 10.0))

hypothesis = tf.mul(X, W)

cost = tf.reduce_sum(tf.square(hypothesis - Y))

decent = W - tf.mul(0.1, tf.reduce_mean(tf.mul((tf.mul(W, X) - Y), X)))
update = W.assign(decent)

init = tf.initialize_all_variables()


sess = tf.Session()
sess.run(init)


for step in xrange(100):
    sess.run(update, feed_dict={X:x_data, Y:y_data})
    print step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W)

