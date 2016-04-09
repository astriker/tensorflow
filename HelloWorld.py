



import tensorflow as tf

def test_hello_tf():
    hello = tf.constant('Hello, TensorFlow!')
    sess = tf.Session()
    print sess.run(hello)

    a = tf.constant(10)
    b = tf.constant(32)
    print sess.run(a + b)





if __name__ == '__main__':

    test_hello_tf()