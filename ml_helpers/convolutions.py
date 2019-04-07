import numpy as np

# Adds zeros around the border of an input
# Allows you to use a conv layer without shrinking the height/width of the volumes
# Helps keep information from the border of the input
def zero_pad(X, pad):
  return np.pad(X, ((0,0), (pad, pad), (pad, pad), (0,0)), 'constant')

def conv_step(a_slice, W, b):
  s = np.multiply(a_slice, W)
  Z = np.sum(s)
  return float(Z + b)

def conv_forward(A_prev, W, b, hparameters):
  (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
  (f, f, n_C_prev, n_C) = W.shape
  stride = hparameters['stride']
  pad = hparameters['pad']

  n_H = int((n_H_prev - f + 2*pad) / stride + 1)
  n_W = int((n_W_prev - f + 2*pad) / stride + 1)

  Z = np.zeros((m, n_H, n_W, n_C))

  A_prev_pad = zero_pad(A_prev, pad)

  # loop through each training example
  for i in range(m):
    a_prev_pad = A_prev_pad[i]
    # loop over the vertical axis of the output volume  
    for h in range(n_H):
      # loop over the horizontal axis of the output volume  
      for w in range(n_W):
        # loop through the channels (= #filters) of the output volume
        for c in range(n_C):
          # corners of current slice
          vert_start = h * stride
          vert_end = vert_start + f
          horiz_start = w * stride
          horiz_end = horiz_start + f

          # use corners to define 3D slice of a_prev_pad
          a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]

          # convolve the 3D slice with filter W and bias b
          Z[i, h, w, c] = conv_step(a_slice_prev, W[:,:,:,c], b[:,:,:,c])

  return Z, (A_prev, W, b, hparameters)

def pool_forward(A_prev, hparameters, mode='max'):
  (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
  
  f = hparameters["f"]
  stride = hparameters["stride"]

  # output dimensions
  n_H = int(1 + (n_H_prev - f) / stride)
  n_W = int(1 + (n_W_prev - f) / stride)
  n_C = n_C_prev

  A = np.zeros((m, n_H, n_W, n_C))

  for i in range(m):
    for h in range(n_H):
      for w in range(n_W):
        for c in range (n_C):
          vert_start = h * stride
          vert_end = vert_start + f
          horiz_start = w * stride
          horiz_end = horiz_start + f

          a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
          
          if mode == 'average':
            A[i, h, w, c] = a_prev_slice.mean()
          elif mode == 'max':  
            A[i, h, w, c] = a_prev_slice.max()

  return A, (A_prev, hparameters)

def conv_backward(dZ, cache):
  (A_prev, W, b, hparameters) = cache
  (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
  (f, f, n_C_prev, n_C) = W.shape
  
  stride = hparameters["stride"]
  pad = hparameters["pad"]

  (m, n_H, n_W, n_C) = dZ.shape

  # initialize
  dA_prev = np.zeros(A_prev.shape)                           
  dW = np.zeros(W.shape)
  db = np.zeros(b.shape)

  # pad A_prev and A_prev_pad
  A_prev_pad = zero_pad(A_prev, pad)
  dA_prev_pad = zero_pad(dA_prev, pad)

  for i in range(m):
    a_prev_pad = A_prev_pad[i]
    da_prev_pad = dA_prev_pad[i]

    for h in range(n_H):
      for w in range(n_W):
        for c in range(n_C):
          vert_start = h * stride
          vert_end = vert_start + f
          horiz_start = w * stride
          horiz_end = horiz_start + f

          a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

          da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i, h, w, c]
          dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
          db[:,:,:,c] += dZ[i, h, w, c]

    dA_prev[i, :, :, :] = da_prev_pad[pad:-pad,pad:-pad,:]
  
  return dA_prev, dW, db

def create_mask_from_window(x):
  return x == np.max(x)

def distribute_value(dz, shape):
  (n_H, n_W) = shape
  average = dz / (n_H * n_W)
  return np.zeros(shape) + average

def pool_backward(dA, cache, mode='max'):
  (A_prev, hparameters) = cache
  
  stride = hparameters["stride"]
  f = hparameters["f"]

  m, n_H_prev, n_W_prev, n_C_prev = A_prev.shape
  m, n_H, n_W, n_C = dA.shape

  dA_prev = np.zeros(A_prev.shape)

  for i in range(m):
    a_prev = A_prev[i]
    for h in range(n_H):
      for w in range(n_W):
        for c in range(n_C):
          vert_start = h * stride
          vert_end = vert_start + f
          horiz_start = w * stride
          horiz_end = horiz_start + f

          if mode == "max":
            # Use the corners and "c" to define the current slice from a_prev (≈1 line)
            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
            # Create the mask from a_prev_slice (≈1 line)
            mask = create_mask_from_window(a_prev_slice)
            # Set dA_prev to be dA_prev + (the mask multiplied by the correct entry of dA) (≈1 line)
            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * dA[i,h,w,c]
          elif mode == 'average':
            # Get the value a from dA (≈1 line)
            da = dA[i, h, w, c]
            # Define the shape of the filter as fxf (≈1 line)
            shape = (f,f)
            # Distribute it to get the correct slice of dA_prev. i.e. Add the distributed value of da. (≈1 line)
            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += distribute_value(da, shape)

  return dA_prev

# tensorflow
import tensorflow as tf

def create_placeholders(n_H0, n_W0, n_C0, n_y):
  # TensorFlow requires that you create placeholders for the input data that will be fed into the model when running the session.
  # Instead of defining the number of training examples, use None

  X = tf.placeholder(tf.float32, (None, n_H0, n_W0, n_C0))
  Y = tf.placeholder(tf.float32, (None, n_y))
  return X, Y

def initialize_tf_parameters(shapes, seed=None):
  if seed:
    tf.set_random_seed(seed)

  parameters = {}

  # number of layers in network
  L = len(shapes)
  
  for l in range(0, L):
    W = 'W{}'.format(str(l+1))
    parameters[W] = tf.get_variable(W, shapes[l], initializer=tf.contrib.layers.xavier_initializer(seed=0))

  return parameters

def tf_forward(X, parameters):
  A = X
  for l in range(0, L):
    W = parameters['W{}'.format(str(l+1))]
    Z = tf.nn.conv2d(tf.cast(A, tf.float32), W, strides=[1,1,1,1], padding="SAME")
    A = tf.nn.relu(Z)
    P1 = tf.nn.max_pool(A, ksize=[1,8,8,1], strides=[1,8,8,1], padding="SAME")