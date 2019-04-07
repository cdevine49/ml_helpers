import numpy as np
import math

def random_mini_batches(X, Y, mini_batch_size = 64, seed=None):
  """
  Creates a list of random minibatches from (X, Y)
  
  Arguments:
  X -- input data, of shape (input size, number of examples)
  Y -- true "label" vector (1 for blue dot / 0 for red dot), of shape (1, number of examples)
  mini_batch_size -- size of the mini-batches, integer
  
  Returns:
  mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
  """

  assert(X.ndim == 2)
  assert(Y.ndim == 2)
  assert(X.shape[1] == Y.shape[1])

  if seed:
    np.random.seed(seed)
      
  (shuffled_X, shuffled_Y) = _shuffle(X, Y)
  return _partition(shuffled_X, shuffled_Y, mini_batch_size)

  
def _shuffle(X, Y):
  m = X.shape[1]  # number of training examples

  permutation = list(np.random.permutation(m))
  shuffled_X = X[:, permutation]
  shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))
  return (shuffled_X, shuffled_Y)

def _partition(shuffled_X, shuffled_Y, mini_batch_size):
  m = shuffled_X.shape[1]
  mini_batches = []
  num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
  
  for k in range(0, num_complete_minibatches):
      mini_batch_X = shuffled_X[:, k*mini_batch_size:(k+1)*mini_batch_size]
      mini_batch_Y = shuffled_Y[:, k*mini_batch_size:(k+1)*mini_batch_size]
      mini_batch = (mini_batch_X, mini_batch_Y)
      mini_batches.append(mini_batch)
  
  # Handling the end case (last mini-batch < mini_batch_size)
  if m % mini_batch_size != 0:
      mini_batch_X = shuffled_X[:, num_complete_minibatches*mini_batch_size:]
      mini_batch_Y = shuffled_Y[:, num_complete_minibatches*mini_batch_size:]
      mini_batch = (mini_batch_X, mini_batch_Y)
      mini_batches.append(mini_batch)

  return mini_batches