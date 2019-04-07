import numpy as np

def sigmoid(Z):
  return np.power(1 + np.exp(-Z), -1)

def sigmoid_derivative(Z):
  S = sigmoid(Z)
  return S * (1 - S)

def relu(Z):
  return np.maximum(0,Z)

def relu_derivative(Z):
  return (Z > 0).astype(int)