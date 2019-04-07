import numpy as np
import tensorflow as tf
try:
  from activations import sigmoid, sigmoid_derivative, relu, relu_derivative
except ImportError:
  from .activations import sigmoid, sigmoid_derivative, relu, relu_derivative


def propagate(parameters, X, Y, activation="sigmoid", regularization="L2", lambd=0, keep_prob=1, forward_prop_seed=None):
  assert(parameters['W1'].shape[1] == X.shape[0])
  assert(X.shape[1] == Y.shape[1])

  A, Ds, caches = _propagate_forward(parameters, X, activation, "sigmoid", keep_prob, forward_prop_seed)
  Y = Y.reshape(A.shape) # ensure Y is the same shape as AL
  
  grads = _propagate_back(A, Y, Ds, caches, regularization, lambd, keep_prob)
  return grads, _cost(A, Y, parameters, regularization, lambd)

def _propagate_forward(parameters, A_prev, activation="relu", output_activation="sigmoid", keep_prob=1, seed=None):
  caches = []
  Ds = [1]
  L = len(parameters) // 2

  if seed:
    np.random.seed(seed)
  
  # Input + hidden layers
  for l in range(1, L):
    W = parameters['W{}'.format(l)]
    b = parameters['b{}'.format(l)]

    A_prev, D, cache = _step_forward(A_prev, W, b, activation, keep_prob)
    caches.append(cache)
    Ds.append(D)

  # Output layer
  W = parameters['W{}'.format(L)]
  b = parameters['b{}'.format(L)]
  AL, D, cache = _step_forward(A_prev, W, b, output_activation, keep_prob=1)
  caches.append(cache)
  return AL, Ds, caches

def _step_forward(A_prev, W, b, activation="sigmoid", keep_prob=1):
  # linear
  # Z[l] = W[l] dot A[l-1] + b[l]
  Z = np.dot(W, A_prev) + b
  linear_cache = (A_prev, W, b)
  
  # activation
  # A[l] = g[l](Z[l])
  if activation == "sigmoid":
    A = sigmoid(Z)
  elif activation == "relu":
    A = relu(Z)
  elif activation == "tanh":
    A = np.tanh(Z)

  D = np.random.rand(*A.shape) < keep_prob
  A *= D
  A /= keep_prob
  
  activation_cache = (Z, activation)

  return A, D, (linear_cache, activation_cache)

def _cost(A, Y, parameters={}, regularization=None, lambd=0):
  m = Y.shape[1]
  logprobs = np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))
  cost = -np.nansum(logprobs) / m
  cost = np.squeeze(cost) # Turns [[17]] to 17

  regularization_cost = 0
  if regularization:
    L = len(parameters) // 2 + 1
    for l in range (1, L):
      if regularization == "L2":
        regularization_cost += np.sum(np.square(parameters["W{}".format(l)]))
    
    regularization_cost *= lambd/(2*m)

  return cost + regularization_cost

def _propagate_back(AL, Y, Ds, caches, regularization, lambd, keep_prob):
  grads = {}
  L = len(caches) # number of layers  

  # derivative of the cost function with respect to AL
  dA = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
  dA = np.nan_to_num(dA)

  # Loop from l=L-1 to l=0
  for l in reversed(range(L)):
    current_cache = caches[l]
    D = Ds[l]
    dw, db = "dW{}".format(l + 1), "db{}".format(l + 1)
    dA, grads[dw], grads[db] = _step_backward(dA, D, current_cache, regularization, lambd, keep_prob)
  
  return grads

def _step_backward(dA, D, cache, regularization, lambd, keep_prob):
  ((A_prev, W, b), (Z, activation)) = cache
  # dZ = dA * g[l]'(Z[l])
  if activation == "relu":
    activation_derivative = relu_derivative(Z)
  elif activation == "sigmoid":
    activation_derivative = sigmoid_derivative(Z)

  dZ = dA * activation_derivative
  m = A_prev.shape[1]
  dA_prev = np.dot(np.transpose(W), dZ) * D / keep_prob
  
  dW = np.divide(np.dot(dZ, np.transpose(A_prev)), m)

  if regularization == "L2":
    dW += (lambd/m) * W

  db = np.divide(np.sum(dZ, axis=1, keepdims=True), m)

  return dA_prev, dW, db

def _tf_propagate_forward(parameters, A_prev, activation="relu", output_activation="sigmoid"):
  L = len(parameters) // 2
  
  # Input + hidden layers
  for l in range(1, L):
    W = parameters['W{}'.format(l)]
    b = parameters['b{}'.format(l)]

    A_prev = _tf_step_forward(A_prev, W, b, activation)

  # Output layer
  W = parameters['W{}'.format(L)]
  b = parameters['b{}'.format(L)]
  return tf.matmul(W, A_prev)  +b

def _tf_step_forward(X, W, b, activation):
  Z = tf.matmul(W, X) + b
  if activation == "relu":
    return tf.nn.relu(Z)

def _tf_cost(Z, Y):
  logits = tf.transpose(Z)
  labels = tf.transpose(Y)

  return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))