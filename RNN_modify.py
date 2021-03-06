# https://www.youtube.com/watch?v=A8wJYfDUYCk
#'worl' -> predict 'orld'

import tensorflow as tf
import numpy as np
from tensorflow.models.rnn import rnn, rnn_cell

char_rdic = ['w', 'o', 'r', 'l', 'd'] #id -> char
char_dic = {w: i for i, w in enumerate(char_rdic)}

x_data = np.array([[1, 0, 0, 0, 0], #w
                  [0, 1, 0, 0, 0], #o
                  [0, 0, 1, 0, 0], #r
                  [0, 0, 0, 1, 0]], #l
                  dtype='f')

sample = [char_dic[c] for c in 'world'] # to index

#Configuration
char_vocab_size = len(char_dic)
rnn_size = char_vocab_size # 1 hot coding (one of 5)
time_step_size = 4 #'worl' -> predict 'orld'
batch_size = 1

#RNN model
rnn_cell = rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = rnn.rnn(rnn_cell, X_split, state)

# logits: list of 2D Tensors of shape [batch_size x num_decoder_symbols]
# targets: list of 1D batch-sized int32 Tensors of the same length as logits
# weights: list of 1D batch-sized float-Tensors of the same length as logits
logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
weights = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [weights])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))
        print (result, [char_rdic[t] for t in result])

