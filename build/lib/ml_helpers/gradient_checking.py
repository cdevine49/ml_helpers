import numpy as np
from functools import reduce
from propagate import _propagate_forward, _cost

def check_gradient(parameters, gradients, X, Y, epsilon=1e-7):
  parameters_values = _dictionary_to_vector(parameters)
  grad = _dictionary_to_vector(gradients, key_bases=["dW", "db"])
  num_parameters = parameters_values.shape[0]
  J_plus = np.zeros((num_parameters, 1))
  J_minus = np.zeros((num_parameters, 1))
  gradapprox = np.zeros((num_parameters, 1))
  for i in range(num_parameters):
    # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
    thetaplus = np.copy(parameters_values)
    thetaplus[i][0] = thetaplus[i][0] + epsilon
    AL = _propagate_forward(_vector_to_dictionary(thetaplus, parameters), X)[0]
    J_plus[i] = _cost(AL, Y)
    
    # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
    thetaminus = np.copy(parameters_values)
    thetaminus[i][0] = thetaminus[i][0] - epsilon
    AL = _propagate_forward(_vector_to_dictionary(thetaminus, parameters), X)[0]
    J_minus[i] = _cost(AL, Y)
    
    # Compute gradapprox[i]
    gradapprox[i] = (J_plus[i] - J_minus[i]) / (2*epsilon)
    
  # Compare gradapprox to backward propagation gradients by computing difference.
  # print(gradapprox)
  numerator = np.linalg.norm(grad - gradapprox)
  denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
  difference = numerator/denominator

  if difference > 2e-7:
      print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
  else:
      print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")
  
  return difference


def _dictionary_to_vector(dictionary, key_bases=["W", "b"]):
  vector = np.zeros((0,1))
  L = len(dictionary) // 2
  for l in range(L):
    for key in key_bases:
      new_vector = np.reshape(dictionary[key + str(l + 1)], (-1, 1))
      vector = np.concatenate((vector, new_vector), axis=0)

  return vector

def _vector_to_dictionary(vector, parameters, key_bases=["W", "b"]):
  result = {}
  L = len(parameters) // 2
  count = 0
  for l in range(L):
    for key in key_bases:
      shape = parameters[key + str(l + 1)].shape
      size = reduce(lambda x, y: x*y, shape)
      result[key + str(l + 1)] = vector[count: count + size].reshape(shape)
      count += size

  return result
