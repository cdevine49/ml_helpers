import tensorflow as tf

def create_placeholders(n_x, n_y):
  X = tf.placeholder(tf.float32, [n_x, None])
  Y = tf.placeholder(tf.float32, [n_y, None])
  return X, Y