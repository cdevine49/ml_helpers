import numpy as np

def vectorize(array, m=1):
  """
  Argument:
  image -- a numpy array of shape (a,b, ..., z)
  
  Returns:
  v -- a vector of shape (a*b* ... *z, 1)
  """

  assert(type(array) is np.ndarray)
  return array.reshape(m, -1).T

def normalize_rows(X):
  """    
  Argument:
  X -- A numpy matrix of shape (n, m)
  
  Returns:
  The numpy matrix X normalized (by row) numpy matrix
  """
  # Each row of X is divided by the sqare root of the squared sums of its elements
  # ||[a,b,c]|| == np.sqrt(a**2 + b**2 + c**2)
  X_norm = np.linalg.norm(X, axis=1, ord=2, keepdims=True)

  # Replace zeros to avoid division error
  X_norm[X_norm == 0] = 1
  return X / X_norm

def softmax(X):
  """
  Calculates the softmax for each row of the input x.

  Your code should work for a row vector and also for matrices of shape (n, m).

  Argument:
  x -- A numpy matrix of shape (n,m)

  Returns:
  s -- A numpy matrix equal to the softmax of x, of shape (n,m)
  """
  
  # Array of elements e raised to power of corresponding element in X
  X_exp = np.exp(X)

  # A vector of the sum of each row of x_exp
  X_sum = np.sum(X_exp, axis=1, keepdims=True)
  
  return X_exp / X_sum

def loss(yhat, y, L=2):
  assert(L in [1,2])
  if L == 1:
    return np.sum(np.absolute(y - yhat))
  elif L == 2:
    return np.sum(np.power(y - yhat, 2))