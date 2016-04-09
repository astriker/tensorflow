import math
import tensorflow as tf
import numpy as np

HIDDEN_NODES = 10

x = tf.placeholder(tf.float32, [None, 2])
W_hidden = tf.Variable(tf.truncated_normal([2, HIDDEN_NODES]))
b_hidden = tf.Variable(tf.zeros([HIDDEN_NODES]))
hidden = tf.nn.relu(tf.matmul(x, W_hidden) + b_hidden)

W_logits = tf.Variable(tf.truncated_normal([HIDDEN_NODES, 1]))
b_logits = tf.Variable(tf.zeros([1]))
logits = tf.add(tf.matmul(hidden, W_logits),b_logits)


y = tf.nn.sigmoid(logits)


y_input = tf.placeholder(tf.float32, [None, 1])



loss = -(y_input * tf.log(y) + (1 - y_input) * tf.log(1 - y))

train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

init_op = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init_op)

xTrain = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])


yTrain = np.array([[0], [1], [1], [0]])


for i in xrange(2000):
  _, loss_val,logitsval = sess.run([train_op, loss,logits], feed_dict={x: xTrain, y_input: yTrain})

  if i % 10 == 0:
    print "Step:", i, "Current loss:", loss_val,"logits",logitsval

print "---------"
print sess.run(y,feed_dict={x: xTrain})