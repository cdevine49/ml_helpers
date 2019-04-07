import numpy as np

def initial_velocity(parameters):
  """
  Initializes the velocity as a python dictionary with:
              - keys: "dW1", "db1", ..., "dWL", "dbL" 
              - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
  Arguments:
  parameters -- python dictionary containing your parameters.
                  parameters['W' + str(l)] = Wl
                  parameters['b' + str(l)] = bl
  
  Returns:
  v -- python dictionary containing the current velocity.
                  v['dW' + str(l)] = velocity of dWl
                  v['db' + str(l)] = velocity of dbl
  """
  
  L = len(parameters) // 2 # number of layers in the neural networks
  v = {}
  
  # Initialize velocity
  for l in range(L):
      v["dW" + str(l+1)] = np.zeros(parameters["W{}".format(l+1)].shape)
      v["db" + str(l+1)] = np.zeros(parameters["b{}".format(l+1)].shape)
      
  return v

def update_parameters(parameters, gradients, v, beta, learning_rate):
  """
  Update parameters using Momentum
  
  Arguments:
  parameters -- python dictionary containing your parameters:
                  parameters['W' + str(l)] = Wl
                  parameters['b' + str(l)] = bl
  gradients -- python dictionary containing your gradients for each parameters:
                  gradients['dW' + str(l)] = dWl
                  gradients['db' + str(l)] = dbl
  v -- python dictionary containing the current velocity:
                  v['dW' + str(l)] = ...
                  v['db' + str(l)] = ...
  beta -- the momentum hyperparameter, scalar
  learning_rate -- the learning rate, scalar
  
  Returns:
  parameters -- python dictionary containing your updated parameters 
  v -- python dictionary containing your updated velocities
  """

  L = len(parameters) // 2 # number of layers in the neural networks

  for l in range(L):    
    # compute velocities
    v["dW" + str(l+1)] = beta*v['dW{}'.format(l+1)] + (1-beta) * gradients['dW{}'.format(l+1)]
    v["db" + str(l+1)] = beta*v['db{}'.format(l+1)] + (1-beta) * gradients['db{}'.format(l+1)]
    # update parameters
    parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v['dW{}'.format(l+1)]
    parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v['db{}'.format(l+1)]
        
  return parameters, v
