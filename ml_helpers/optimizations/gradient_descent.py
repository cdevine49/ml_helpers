import numpy as np

def gradient_descent(parameters, gradients, alpha=0.1):
  result = {}
  L = len(parameters) // 2
  for l in range(L):
    W = "W{}".format(str(l+1))
    b = "b{}".format(str(l+1))
    result[W] = parameters[W] - alpha * gradients['d{}'.format(W)]
    result[b] = parameters[b] - alpha * gradients['d{}'.format(b)]

  return result