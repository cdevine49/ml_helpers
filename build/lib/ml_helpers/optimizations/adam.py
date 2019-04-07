import numpy as np

def initialize(parameters) :
  """
  Initializes v and s as two python dictionaries with:
              - keys: "dW1", "db1", ..., "dWL", "dbL" 
              - values: numpy arrays of zeros of the same shape as the corresponding gradients/parameters.
  
  Arguments:
  parameters -- python dictionary containing your parameters.
                  parameters["W" + str(l)] = Wl
                  parameters["b" + str(l)] = bl
  
  Returns: 
  v -- python dictionary that will contain the exponentially weighted average of the gradient.
                  v["dW" + str(l)] = ...
                  v["db" + str(l)] = ...
  s -- python dictionary that will contain the exponentially weighted average of the squared gradient.
                  s["dW" + str(l)] = ...
                  s["db" + str(l)] = ...

  """
  
  L = len(parameters) // 2 # number of layers in the neural networks
  v = {}
  s = {}
  
  # Initialize v, s. Input: "parameters". Outputs: "v, s".
  for l in range(L):
      v["dW" + str(l+1)] = np.zeros(parameters["W{}".format(l+1)].shape)
      v["db" + str(l+1)] = np.zeros(parameters["b{}".format(l+1)].shape)
      s["dW" + str(l+1)] = np.zeros(parameters["W{}".format(l+1)].shape)
      s["db" + str(l+1)] = np.zeros(parameters["b{}".format(l+1)].shape)
  
  return v, s

def update_parameters(parameters, grads, v, s, t, learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999,  epsilon = 1e-8):
    """
    Update parameters using Adam
    
    Arguments:
    parameters -- python dictionary containing your parameters:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients for each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    learning_rate -- the learning rate, scalar.
    beta1 -- Exponential decay hyperparameter for the first moment estimates 
    beta2 -- Exponential decay hyperparameter for the second moment estimates 
    epsilon -- hyperparameter preventing division by zero in Adam updates

    Returns:
    parameters -- python dictionary containing your updated parameters 
    v -- Adam variable, moving average of the first gradient, python dictionary
    s -- Adam variable, moving average of the squared gradient, python dictionary
    """
    
    L = len(parameters) // 2                 # number of layers in the neural networks
    v_corrected = {}                         # Initializing first moment estimate, python dictionary
    s_corrected = {}                         # Initializing second moment estimate, python dictionary
    
    # Perform Adam update on all parameters
    for l in range(L):
      dWl = "dW{}".format(l+1)
      dbl = "db{}".format(l+1)

      # Moving average of the gradients. Inputs: "v, grads, beta1". Output: "v".
      v[dWl] = beta1 * v[dWl] + (1-beta1) * grads[dWl]
      v[dbl] = beta1 * v[dbl] + (1-beta1) * grads[dbl]

      # Compute bias-corrected first moment estimate. Inputs: "v, beta1, t". Output: "v_corrected".
      v_corrected[dWl] = v[dWl] / (1-np.power(beta1,t))
      v_corrected[dbl] = v[dbl] / (1-np.power(beta1,t))

      # Moving average of the squared gradients. Inputs: "s, grads, beta2". Output: "s".
      s[dWl] = beta2 * s[dWl] + (1-beta2) * np.power(grads[dWl],2)
      s[dbl] = beta2 * s[dbl] + (1-beta2) * np.power(grads[dbl],2)

      # Compute bias-corrected second raw moment estimate. Inputs: "s, beta2, t". Output: "s_corrected".
      s_corrected[dWl] = s[dWl] / (1-np.power(beta2,t))
      s_corrected[dbl] = s[dbl] / (1-np.power(beta2,t))

      # Update parameters. Inputs: "parameters, learning_rate, v_corrected, s_corrected, epsilon". Output: "parameters".
      parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * v_corrected[dWl] / (np.sqrt(s_corrected[dWl]) + epsilon)
      parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * v_corrected[dbl] / (np.sqrt(s_corrected[dbl]) + epsilon)

    return parameters, v, s