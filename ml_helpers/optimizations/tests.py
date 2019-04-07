import unittest
import numpy as np

from momentum import initial_velocity, update_parameters as update_momentum_parameters
from random_mini_batches import random_mini_batches
from adam import initialize as initialize_adam, update_parameters as update_adam_parameters
from gradient_descent import gradient_descent

class TestMomentum(unittest.TestCase):

  def test_initial_velocity(self):
    self.assertEqual(initial_velocity({}), {})
    
    parameters = {  
      'W1': np.array(
        [
          [1.62434536, -0.61175641, -0.52817175],
          [-1.07296862, 0.86540763, -2.3015387]
        ]
      ),
      'b1': np.array([[1.74481176],[-0.7612069]]),
      'W2': np.array(
        [
          [0.3190391, -0.24937038, 1.46210794],
          [-2.06014071, -0.3224172, -0.38405435],
          [1.13376944, -1.09989127, -0.17242821]
        ]
      ),
      'b2': np.array([[-0.87785842], [0.04221375], [0.58281521]])
    }
    expected = {
      'dW1': np.array(
        [
          [0., 0., 0.],
          [0., 0., 0.]
        ]
      ), 
      'db1': np.array([[0.], [0.]]), 
      'dW2': np.array(
        [
          [0., 0., 0.],
          [0., 0., 0.],
          [0., 0., 0.]
        ]
      ), 
      'db2': np.array([[0.], [0.], [0.]])
    }

    res = initial_velocity(parameters)
    
    self.assertTrue(np.array_equal(res['dW1'], expected['dW1']))
    self.assertTrue(np.array_equal(res['db1'], expected['db1']))
    self.assertTrue(np.array_equal(res['dW2'], expected['dW2']))
    self.assertTrue(np.array_equal(res['db2'], expected['db2']))

  def test_update_parameters(self):
    self.assertEqual(update_momentum_parameters({}, {}, {}, None, None), ({}, {}))

    parameters = {
      'W1': np.array([
        [1.62434536, -0.61175641, -0.52817175],
        [-1.07296862, 0.86540763, -2.3015387 ]
      ]),
      'b1': np.array([
        [1.74481176],
        [-0.7612069 ]
      ]),
      'W2': np.array([
        [0.3190391, -0.24937038, 1.46210794],
        [-2.06014071, -0.3224172, -0.38405435],
        [1.13376944, -1.09989127, -0.17242821]
      ]),
      'b2': np.array([
        [-0.87785842],
        [0.04221375],
        [0.58281521]
      ])
    }

    grads = {
      'dW1': np.array([
        [-1.10061918, 1.14472371, 0.90159072],
        [0.50249434, 0.90085595, -0.68372786]
      ]), 
      'db1': np.array([
        [-0.12289023],
        [-0.93576943]
      ]),
      'dW2': np.array([
        [-0.26788808, 0.53035547, -0.69166075],
        [-0.39675353, -0.6871727, -0.84520564],
        [-0.67124613, -0.0126646, -1.11731035]
      ]),
      'db2': np.array([
        [0.2344157],
        [1.65980218],
        [0.74204416]
      ])
    }

    v = {
      'dW1': np.array([
        [0, 0, 0],
        [0, 0, 0]
      ]), 
      'db1': np.array([
        [0],
        [0]
      ]),
      'dW2': np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 0]
      ]),
      'db2': np.array([
        [0],
        [0],
        [0]
      ])
    }

    expected = {
      'parameters': {
        'W1': np.array([
          [1.62544598, -0.61290113, -0.52907334],
          [-1.07347111, 0.86450677, -2.30085497]
        ]),
        'b1': np.array([
          [1.74493465],
          [-0.76027113]
        ]),
        'W2': np.array([
          [0.31930699, -0.24990074, 1.4627996],
          [-2.05974396, -0.32173003, -0.38320914],
          [1.13444069, -1.09987861, -0.1713109]
        ]),
        'b2': np.array([
          [-0.87809283],
          [0.04055394],
          [0.58207317]
        ])
      },
      'v': {
        'dW1': np.array([
          [-0.11006192, 0.11447237, 0.09015907],
          [0.05024943, 0.09008559, -0.06837279]
        ]),
        'db1': np.array([
          [-0.01228902],
          [-0.09357694]
        ]),
        'dW2': np.array([
          [-0.02678881, 0.05303555, -0.06916608],
          [-0.03967535, -0.06871727, -0.08452056],
          [-0.06712461, -0.00126646, -0.11173103]
        ]),
        'db2': np.array([
          [0.02344157],
          [0.16598022],
          [0.07420442]
        ])
      }
    }

    res_parameters, res_v = update_momentum_parameters(parameters, grads, v, 0.9, 0.01)

    self.assertTrue(np.allclose(res_parameters['W1'], expected['parameters']['W1']))
    self.assertTrue(np.allclose(res_parameters['b1'], expected['parameters']['b1']))
    self.assertTrue(np.allclose(res_parameters['W2'], expected['parameters']['W2']))
    self.assertTrue(np.allclose(res_parameters['b2'], expected['parameters']['b2']))
    
    self.assertTrue(np.allclose(res_v['dW1'], expected['v']['dW1']))
    self.assertTrue(np.allclose(res_v['db1'], expected['v']['db1']))
    self.assertTrue(np.allclose(res_v['dW2'], expected['v']['dW2']))
    self.assertTrue(np.allclose(res_v['db2'], expected['v']['db2']))

    
class TestRandomMiniBatches(unittest.TestCase):

  def test_random_mini_batches(self):
    # X must be multidimensional
    with self.assertRaises(AssertionError):
      random_mini_batches(np.array([]), np.array([[]]))

    # Y must be multidimensional
    with self.assertRaises(AssertionError):
      random_mini_batches(np.array([[]]), np.array([]))

    # each example X must have a result Y
    with self.assertRaises(AssertionError):
      random_mini_batches(np.array([[1,2,3], [4,5,6]]), np.array([[0,1]]))

    
    size = 200
    r = np.arange(0,size)
    X = np.array([r,r,r])
    Y = np.array([r % 2 == 0])

    # mini batch size defaults to 64
    # each mini batch contains correctly sorted X and Y values
    mini_batches = random_mini_batches(X, Y)
    self.assertTrue(np.shape(mini_batches) == (4,2))
    L = len(mini_batches) - 1
    for l in range(0, L):
      current_batch = mini_batches[l]
      self.assertTrue(len(current_batch), 64)
      self.assertTrue(np.array_equal(current_batch[0][0], current_batch[0][1]))
      self.assertTrue(np.array_equal(current_batch[0][0], current_batch[0][2]))
      self.assertTrue(((current_batch[0][0] % 2 == 0) == current_batch[1][0]).all())
   

    current_batch = mini_batches[L]

    self.assertTrue(len(current_batch), 8)
    self.assertTrue(np.array_equal(current_batch[0][0], current_batch[0][1]))
    self.assertTrue(np.array_equal(current_batch[0][0], current_batch[0][2]))
    self.assertTrue(((current_batch[0][0] % 2 == 0) == current_batch[1][0]).all())

    # with mini batch size specified
    mini_batches = random_mini_batches(X, Y, 99)
    self.assertTrue(np.shape(mini_batches) == (3,2))
    L = len(mini_batches) - 1
    for l in range(0, L):
      current_batch = mini_batches[l]
      self.assertTrue(len(current_batch), 64)
      self.assertTrue(np.array_equal(current_batch[0][0], current_batch[0][1]))
      self.assertTrue(np.array_equal(current_batch[0][0], current_batch[0][2]))
      self.assertTrue(((current_batch[0][0] % 2 == 0) == current_batch[1][0]).all())
   

    current_batch = mini_batches[L]

    self.assertTrue(len(current_batch), 2)
    self.assertTrue(np.array_equal(current_batch[0][0], current_batch[0][1]))
    self.assertTrue(np.array_equal(current_batch[0][0], current_batch[0][2]))
    self.assertTrue(((current_batch[0][0] % 2 == 0) == current_batch[1][0]).all())


class TestAdam(unittest.TestCase):
  def test_initialize_adaam(self):
    self.assertEqual(initialize_adam({}), ({}, {}))

    parameters = {
      'W1': np.array([
        [1.62434536, -0.61175641, -0.52817175],
        [-1.07296862, 0.86540763, -2.3015387 ]
      ]),
      'b1': np.array([
        [1.74481176],
        [-0.7612069]
      ]),
      'W2': np.array([
        [0.3190391, -0.24937038, 1.46210794],
        [-2.06014071, -0.3224172, -0.38405435],
        [1.13376944, -1.09989127, -0.17242821]
      ]),
      'b2': np.array([[-0.87785842],
        [0.04221375],
        [0.58281521]
      ])
    }

    expected = {
      'v': {
        'dW1': np.array([
          [0, 0, 0],
          [0, 0, 0]
        ]),
        'db1': np.array([
          [0],
          [0]
        ]),
        'dW2': np.array([
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]
        ]),
        'db2': np.array([
          [0],
          [0],
          [0]
        ])
      },
      's': {
        'dW1': np.array([
          [0, 0, 0],
          [0, 0, 0]
        ]),
        'db1': np.array([
          [0],
          [0]
        ]),
        'dW2': np.array([
          [0, 0, 0],
          [0, 0, 0],
          [0, 0, 0]
        ]),
        'db2': np.array([
          [0],
          [0],
          [0]
        ])
      }
    }

    v, s = initialize_adam(parameters)
    self.assertTrue(np.allclose(v['dW1'], expected['v']['dW1']))
    self.assertTrue(np.allclose(v['db1'], expected['v']['db1']))
    self.assertTrue(np.allclose(v['dW2'], expected['v']['dW2']))
    self.assertTrue(np.allclose(v['db2'], expected['v']['db2']))
    
    self.assertTrue(np.allclose(s['dW1'], expected['s']['dW1']))
    self.assertTrue(np.allclose(s['db1'], expected['s']['db1']))
    self.assertTrue(np.allclose(s['dW2'], expected['s']['dW2']))
    self.assertTrue(np.allclose(s['db2'], expected['s']['db2']))

  def test_update_parameters(self):
    self.assertEqual(update_adam_parameters({},{},{},{}, {}), ({},{},{}))
    
    v = {'a': 'b'}
    s = {'c': 'd'}
    self.assertEqual(update_adam_parameters({},{},v,s, {}), ({},v,s))

    parameters = {
      'W1': np.array([
        [1.62434536, -0.61175641, -0.52817175],
        [-1.07296862, 0.86540763, -2.3015387 ]
      ]),
      'b1': np.array([
        [1.74481176],
        [-0.7612069]
      ]),
      'W2': np.array([
        [0.3190391, -0.24937038, 1.46210794],
        [-2.06014071, -0.3224172, -0.38405435],
        [1.13376944, -1.09989127, -0.17242821]
      ]),
      'b2': np.array([
        [-0.87785842],
        [0.04221375],
        [0.58281521]
      ])
    }

    grads = {
      'dW1': np.array([
        [-1.10061918, 1.14472371, 0.90159072],
        [0.50249434, 0.90085595, -0.68372786]
      ]),
      'db1': np.array([
        [-0.12289023],
        [-0.93576943]
      ]),
      'dW2': np.array([
        [-0.26788808, 0.53035547, -0.69166075],
        [-0.39675353, -0.6871727, -0.84520564],
        [-0.67124613, -0.0126646, -1.11731035]
      ]),
      'db2': np.array([
        [0.2344157],
        [1.65980218],
        [0.74204416]
      ])
    }

    v, s = initialize_adam(parameters)
    t = 2

    expected = {
      'parameters': {
        'W1': np.array([
          [1.63178673, -0.61919778, -0.53561312],
          [-1.08040999, 0.85796626, -2.29409733]
        ]),
        'b1': np.array([
          [1.75225313],
          [-0.75376553]
        ]),
        'W2': np.array([
          [0.32648046, -0.25681174, 1.46954931],
          [-2.05269934, -0.31497584, -0.37661299],
          [1.14121081, -1.09244991, -0.16498684]
        ]),
        'b2': np.array([
          [-0.88529979],
          [0.03477238],
          [0.57537385]
        ])
      },
      'v': {
        'dW1': np.array([
          [-0.11006192, 0.11447237, 0.09015907],
          [0.05024943, 0.09008559, -0.06837279]
        ]),
        'db1': np.array([
          [-0.01228902],
          [-0.09357694]
        ]),
        'dW2': np.array([
          [-0.02678881, 0.05303555, -0.06916608],
          [-0.03967535, -0.06871727, -0.08452056],
          [-0.06712461, -0.00126646, -0.11173103]
        ]),
        'db2': np.array([
          [0.02344157],
          [0.16598022],
          [0.07420442]
        ])
      },
      's': {
        'dW1': np.array([
          [0.00121136, 0.00131039, 0.00081287],
          [0.0002525, 0.00081154, 0.00046748]
        ]),
        'db1': np.array([
          [1.51020075e-05],
          [8.75664434e-04]
        ]),
        'dW2': np.array([
          [7.17640232e-05, 2.81276921e-04, 4.78394595e-04],
          [1.57413361e-04, 4.72206320e-04, 7.14372576e-04],
          [4.50571368e-04, 1.60392066e-07, 1.24838242e-03]
        ]),
        'db2': np.array([
          [5.49507194e-05],
          [2.75494327e-03],
          [5.50629536e-04]
        ])
      }
    }

    res_parameters, res_v, res_s = update_adam_parameters(parameters, grads, v, s, t)
    self.assertTrue(np.allclose(res_parameters['W1'], expected['parameters']['W1']))
    self.assertTrue(np.allclose(res_parameters['b1'], expected['parameters']['b1']))
    self.assertTrue(np.allclose(res_parameters['W2'], expected['parameters']['W2']))
    self.assertTrue(np.allclose(res_parameters['b2'], expected['parameters']['b2']))
    self.assertTrue(np.allclose(res_v['dW1'], expected['v']['dW1']))
    self.assertTrue(np.allclose(res_v['db1'], expected['v']['db1']))
    self.assertTrue(np.allclose(res_v['dW2'], expected['v']['dW2']))
    self.assertTrue(np.allclose(res_v['db2'], expected['v']['db2']))
    self.assertTrue(np.allclose(res_s['dW1'], expected['s']['dW1']))
    self.assertTrue(np.allclose(res_s['db1'], expected['s']['db1']))
    self.assertTrue(np.allclose(res_s['dW2'], expected['s']['dW2']))
    self.assertTrue(np.allclose(res_s['db2'], expected['s']['db2']))


class TestGradientDescent(unittest.TestCase):
  def test_gradient_descent(self):
    parameters = {
      'W1': np.array([[0], [0]]),
      'b1': 0
    }

    gradients = {
      'dW1': np.array([[0],[0]]),
      'db1': 0
    }

    updated_parameters = gradient_descent(parameters, gradients)
    self.assertTrue(np.array_equal(updated_parameters['W1'], np.zeros((2,1))))
    self.assertEqual(updated_parameters['b1'], 0)

    parameters = {
      'W1': np.array([[2], [3]]),
      'b1': 4
    }

    gradients = {
      'dW1': np.array([[1],[5]]),
      'db1': 6
    }

    updated_parameters = gradient_descent(parameters, gradients)
    self.assertTrue(np.array_equal(updated_parameters['W1'], np.array([[1.9], [2.5]])))
    self.assertEqual(updated_parameters['b1'], 3.4)

    updated_parameters = gradient_descent(parameters, gradients, alpha=0.3)
    self.assertTrue(np.array_equal(updated_parameters['W1'], np.array([[1.7], [1.5]])))
    self.assertEqual(updated_parameters['b1'], 2.2)

    parameters = {
      'W1': np.array([
        [1.62434536, -0.61175641, -0.52817175],
        [-1.07296862, 0.86540763, -2.3015387]
      ]),
      'b1': np.array([[1.74481176], [-0.7612069]]),
      'W2': np.array([
        [0.3190391, -0.24937038, 1.46210794],
        [-2.06014071, -0.3224172, -0.38405435],
        [ 1.13376944, -1.09989127, -0.17242821]]),
      'b2': np.array([[-0.87785842], [0.04221375], [0.58281521]])
    }

    gradients = {
      'dW1': np.array([
        [-1.10061918, 1.14472371, 0.90159072],
        [ 0.50249434, 0.90085595, -0.68372786]
      ]),
      'db1': np.array([[-0.12289023], [-0.93576943]]),
      'dW2': np.array([
        [-0.26788808, 0.53035547, -0.69166075],
        [-0.39675353, -0.6871727, -0.84520564],
        [-0.67124613, -0.0126646, -1.11731035]]),
      'db2': np.array([[0.2344157], [1.65980218], [0.74204416]])
    }

    expected = {
      'W1': np.array([
        [1.63535156, -0.62320365, -0.53718766],
        [-1.07799357, 0.85639907, -2.29470142]
      ]),
      'b1': np.array([[1.74604067], [-0.75184921]]),
      'W2': np.array([
        [ 0.32171798, -0.25467393, 1.46902454],
        [-2.05617317, -0.31554548, -0.3756023],
        [ 1.1404819, -1.09976462, -0.1612551]
      ]),
      'b2': np.array([[-0.88020257], [ 0.02561572], [ 0.57539477]])
    }

    updated_parameters = gradient_descent(parameters, gradients, 0.01)
    self.assertTrue(np.allclose(updated_parameters['W1'], expected['W1']))
    self.assertTrue(np.allclose(updated_parameters['b1'], expected['b1']))
    self.assertTrue(np.allclose(updated_parameters['W2'], expected['W2']))
    self.assertTrue(np.allclose(updated_parameters['b2'], expected['b2']))


if __name__ == '__main__':
  unittest.main()