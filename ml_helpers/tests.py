import unittest
import numpy as np
from activations import sigmoid, sigmoid_derivative, relu, relu_derivative
from propagate import propagate, _cost, _step_forward, _propagate_forward, _step_backward, _propagate_back, _tf_propagate_forward, _tf_cost
from utils import vectorize, normalize_rows, softmax, loss
from initializations import initialize_parameters, initialize_lr_with_zeros, initialize_tf_parameters
from gradient_checking import _dictionary_to_vector, check_gradient
from convolutions import zero_pad, conv_step, conv_forward, pool_forward, conv_backward, create_mask_from_window, distribute_value, pool_backward, create_placeholders, initialize_tf_parameters as initialize_tf_conv_parameters

import tensorflow as tf
from tf_helpers import create_placeholders


class TestActivations(unittest.TestCase):

  def test_sigmoid(self):
    self.assertEqual(sigmoid(0), 0.5)
    self.assertGreater(sigmoid(100), .99)
    self.assertLess(sigmoid(-100), .01)

    Z = np.array([1,2,3])
    expected = np.array([0.73105858, 0.88079708, 0.95257413])
    self.assertTrue(np.allclose(sigmoid(Z), expected))

  def test_sigmoid_derivative(self):
    self.assertEqual(sigmoid_derivative(0), 0.25)
    self.assertLess(sigmoid_derivative(100), .001)
    self.assertLess(sigmoid_derivative(-100), .001)

    Z = np.array([1, 2, 3])
    expected = np.array([0.1966119, 0.1049935, 0.04517666])
    self.assertTrue(np.allclose(sigmoid_derivative(Z), expected))

  def test_relu(self):
    self.assertEqual(relu(1), 1)
    self.assertEqual(relu(0), 0)
    self.assertEqual(relu(-1), 0)

  def test_relu_derivative(self):
    self.assertTrue(np.allclose(relu_derivative(np.array([[2,1,0,-1,-2]])), np.array([[1,1,0,0,0]])))

class TestUtils(unittest.TestCase):
  
  def test_vectorize(self):
    with self.assertRaises(AssertionError):
      vectorize([])
    
    array = vectorize(np.array([]))
    self.assertEqual(array.shape, (0,1))

    array = vectorize(np.zeros((2,1)))
    self.assertEqual(array.shape, (2,1))

    array = vectorize(np.zeros((1,3)))
    self.assertEqual(array.shape, (3,1))

    array = vectorize(np.zeros((3,4,2,5)))
    self.assertEqual(array.shape, (120,1))

    image = np.array([
      [
        [0.67826139, 0.29380381],
        [0.90714982, 0.52835647],
        [0.4215251 , 0.45017551]
      ],
      [
        [0.92814219, 0.96677647],
        [0.85304703, 0.52351845],
        [0.19981397, 0.27417313]
      ],
      [
        [0.60659855, 0.00533165],
        [0.10820313, 0.49978937],
        [0.34144279, 0.94630077]
      ]
    ])

    expected = np.array([
      [0.67826139],
      [0.29380381],
      [0.90714982],
      [0.52835647],
      [0.4215251 ],
      [0.45017551],
      [0.92814219],
      [0.96677647],
      [0.85304703],
      [0.52351845],
      [0.19981397],
      [0.27417313],
      [0.60659855],
      [0.00533165],
      [0.10820313],
      [0.49978937],
      [0.34144279],
      [0.94630077]
    ])

    self.assertTrue(np.allclose(vectorize(image), expected))

  def test_normalize_rows(self):
    arg = np.zeros((3,2))
    self.assertTrue(np.array_equal(normalize_rows(arg), arg))

    arg = np.array([
      [0, 3, 4],
      [1, 6, 4]
    ])

    expected = np.array([
      [0, 0.6, 0.8],
      [0.13736056, 0.82416338, 0.54944226]
    ])

    self.assertTrue(np.allclose(normalize_rows(arg), expected))

  def test_softmax(self):
    arg = np.zeros((3,2))
    self.assertTrue(np.array_equal(softmax(arg), arg + 0.5))

    arg = np.array([
      [9, 2, 5, 0, 0],
      [7, 5, 0, 0 ,0]
    ])

    expected = np.array([
      [9.80897665e-01, 8.94462891e-04, 1.79657674e-02, 1.21052389e-04, 1.21052389e-04],
      [8.78679856e-01, 1.18916387e-01, 8.01252314e-04, 8.01252314e-04, 8.01252314e-04]
    ])

    self.assertTrue(np.allclose(softmax(arg), expected))

  def test_loss(self):
    with self.assertRaises(AssertionError):
      loss([], [], L=0)
    
    with self.assertRaises(AssertionError):
      loss([], [], L=3)

    size = 10
    y = np.random.randint(2, size=size)
    yhat = np.copy(y)

    self.assertEqual(loss(yhat, y, L=1), 0)
    self.assertEqual(loss(yhat, y, L=2), 0)

    yhat = (y == 0).astype(int)
    self.assertEqual(loss(yhat, y, L=1), size)
    self.assertEqual(loss(yhat, y, L=2), size)

    y = np.array([1, 0, 0, 1, 1])
    yhat = np.array([.9, 0.2, 0.1, .4, .9])

    self.assertEqual(loss(yhat, y, L=1), 1.1)
    self.assertEqual(loss(yhat, y, L=2), 0.43)

class TestInitialization(unittest.TestCase):

  def test_initialize_parameters(self):
    for i in range(5, 8):
      for o in range(1, 4):
        layer_dimensions = [dimension for dimension in [i,o] if dimension != 0]
        parameters = initialize_parameters(layer_dimensions, use="zeros")
        self.assertEqual(len(parameters), 2)
        
        self.assertTrue(np.array_equal(parameters["W1"], np.zeros((o,i))))
        self.assertTrue(np.array_equal(parameters["b1"], np.zeros((o, 1))))
        for h1 in range(1, 4):
          layer_dimensions = [dimension for dimension in [i, h1, o] if dimension != 0]
          parameters = initialize_parameters(layer_dimensions, use="zeros")
          self.assertEqual(len(parameters), 4)
          
          self.assertTrue(np.array_equal(parameters["W1"], np.zeros((h1,i))))
          self.assertTrue(np.array_equal(parameters["b1"], np.zeros((h1, 1))))
          
          self.assertTrue(np.array_equal(parameters["W2"], np.zeros((o, h1))))
          self.assertTrue(np.array_equal(parameters["b2"], np.zeros((o, 1))))
          for h2 in range(7, 9):
            layer_dimensions = [dimension for dimension in [i,h1,h2,o] if dimension != 0]
            parameters = initialize_parameters(layer_dimensions, use="zeros")
            self.assertEqual(len(parameters), 6)
            
            self.assertTrue(np.array_equal(parameters["W1"], np.zeros((h1,i))))
            self.assertTrue(np.array_equal(parameters["b1"], np.zeros((h1, 1))))
            
            self.assertTrue(np.array_equal(parameters["W2"], np.zeros((h2, h1))))
            self.assertTrue(np.array_equal(parameters["b2"], np.zeros((h2, 1))))
            
            self.assertTrue(np.array_equal(parameters["W3"], np.zeros((o, h2))))
            self.assertTrue(np.array_equal(parameters["b3"], np.zeros((o, 1))))

    np.random.seed(3)
    expected = {
      'W1': np.array([
        [17.88628473, 4.36509851, 0.96497468],
        [-18.63492703, -2.77388203, -3.54758979]
      ]),
      'b1': np.array([[0], [0]]),
      'W2': np.array([[-0.82741481, -6.27000677]]),
      'b2': np.array([[0]])
    }

    parameters = initialize_parameters([3,2,1], use="random", W_multiplier=10)
    
    self.assertTrue(np.allclose(parameters["W1"], expected['W1']))
    self.assertTrue(np.allclose(parameters["b1"], expected['b1']))
    self.assertTrue(np.allclose(parameters["W2"], expected['W2']))
    self.assertTrue(np.allclose(parameters["b2"], expected['b2']))

    np.random.seed(3)
    expected = {
      'W1': np.array([
        [1.78862847, 0.43650985],
        [0.09649747, -1.8634927],
        [-0.2773882, -0.35475898],
        [-0.08274148, -0.62700068]
      ]),
      'b1': np.array([
        [0.],
        [0.],
        [0.],
        [0.]
      ]),
      'W2': np.array([[-0.03098412, -0.33744411, -0.92904268, 0.62552248]]),
      'b2': np.array([[0]])
    }

    parameters = initialize_parameters([2,4,1], use="he")
    
    self.assertTrue(np.allclose(parameters["W1"], expected['W1']))
    self.assertTrue(np.allclose(parameters["b1"], expected['b1']))
    self.assertTrue(np.allclose(parameters["W2"], expected['W2']))
    self.assertTrue(np.allclose(parameters["b2"], expected['b2']))
    

  def test_initialize_lr_with_zeros(self):
    with self.assertRaises(ValueError):
      initialize_lr_with_zeros(0)

    for l in range(1,4):
      parameters = initialize_lr_with_zeros(l)
      self.assertTrue(np.array_equal(parameters['W1'], np.zeros((1,l))))
      self.assertEqual(parameters['b1'], 0)
      self.assertEqual(parameters['b1'].shape, (1, 1))

  def test_initialize_tf_parameters(self):
    parameters = initialize_tf_parameters([12288, 25, 12, 6], seed=1)
    tf.get_variable_scope().reuse_variables()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      parameters = sess.run(parameters)
      
    self.assertEqual(parameters['W1'].shape, (25, 12288))
    self.assertAlmostEqual(parameters['W1'][0][0], -0.01962241)
    self.assertEqual(parameters['b1'].shape, (25, 1))
    self.assertEqual(np.sum(parameters['b1']), 0)

    self.assertEqual(parameters['W2'].shape, (12, 25))
    self.assertAlmostEqual(parameters['W2'][0][0], -0.35795909)
    self.assertEqual(parameters['b2'].shape, (12, 1))
    self.assertEqual(np.sum(parameters['b2']), 0)

    self.assertEqual(parameters['W3'].shape, (6, 12))
    self.assertAlmostEqual(parameters['W3'][0][0], -0.5132134)
    self.assertEqual(parameters['b3'].shape, (6, 1))
    self.assertEqual(np.sum(parameters['b3']), 0)

class TestPropagate(unittest.TestCase):

  def test__step_forward(self):
    A_prev = np.array([
      [-0.41675785, -0.05626683],
      [-2.1361961, 1.64027081],
      [-1.79343559, -0.84174737]
    ])
    W = np.array([[0.50288142, -1.24528809, -1.05795222]])
    b = np.array([[-0.90900761]])

    activation = 'sigmoid'
    A, D, ((cA_prev, cW, cb), (cZ, cActivation)) = _step_forward(A_prev, W, b, activation)
    self.assertTrue(np.allclose(A, [[0.96890023, 0.11013289]]))
    self.assertIs(A_prev, cA_prev)
    self.assertIs(W, cW)
    self.assertIs(b, cb)
    self.assertTrue(np.allclose(cZ, np.array([[3.43896134, -2.08938436]])))
    self.assertIs(activation, cActivation)

    activation = 'relu'
    A, D, ((cA_prev, cW, cb), (cZ, cActivation)) = _step_forward(A_prev, W, b, activation)
    self.assertTrue(np.allclose(A, [[3.43896131, 0.]]))
    self.assertIs(A_prev, cA_prev)
    self.assertIs(W, cW)
    self.assertIs(b, cb)
    self.assertTrue(np.allclose(cZ, np.array([[3.43896131, -2.08938436]])))
    self.assertIs(activation, cActivation)

  def test__propagate_forward(self):
    X = np.array([
      [-0.31178367, 0.72900392, 0.21782079, -0.8990918 ],
      [-2.48678065, 0.91325152, 1.12706373, -1.51409323],
      [1.63929108, -0.4298936, 2.63128056, 0.60182225],
      [-0.33588161, 1.23773784, 0.11112817, 0.12915125],
      [0.07612761, -0.15512816, 0.63422534, 0.810655]
    ])

    parameters = {
      'W1': np.array([
        [0.35480861, 1.81259031, -1.3564758 , -0.46363197, 0.82465384],
        [-1.17643148, 1.56448966, 0.71270509, -0.1810066 , 0.53419953],
        [-0.58661296, -1.48185327, 0.85724762, 0.94309899, 0.11444143],
        [-0.02195668, -2.12714455, -0.83440747, -0.46550831, 0.23371059]
      ]),
      'b1': np.array([
        [1.38503523],
        [-0.51962709],
        [-0.78015214],
        [0.95560959]
      ]),
      'W2': np.array([
        [-0.12673638, -1.36861282, 1.21848065, -0.85750144],
        [-0.56147088, -1.0335199 , 0.35877096, 1.07368134],
        [-0.37550472, 0.39636757, -0.47144628, 2.33660781]
      ]),
      'b2': np.array([
        [1.50278553],
        [-0.59545972],
        [0.52834106]
      ]),
      'W3': np.array([
        [0.9398248, 0.42628539, -0.75815703]
      ]),
      'b3': np.array([[-0.16236698]])
    }

    AL, Ds, caches = _propagate_forward(parameters, X)
    self.assertTrue(np.allclose(AL, np.array([[0.03921668, 0.70498921, 0.19734387, 0.04728177]])))
    self.assertEqual(len(caches), 3)

    
    X = np.array([
      [ 1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
      [-2.3015387, 1.74481176, -0.7612069,  0.3190391, -0.24937038],
      [ 1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944]
    ])

    parameters = {
      'W1': np.array([
        [-1.09989127, -0.17242821, -0.87785842],
        [ 0.04221375,  0.58281521, -1.10061918]
      ]),
      'b1': np.array([[ 1.14472371], [ 0.90159072]]),
      'W2': np.array([
        [ 0.50249434,  0.90085595],
        [-0.68372786, -0.12289023],
        [-0.93576943, -0.26788808]
      ]),
      'b2': np.array([[ 0.53035547], [-0.69166075], [-0.39675353]]),
      'W3': np.array([[-0.6871727 , -0.84520564, -0.67124613]]),
      'b3': np.array([[-0.0126646]])
    }
    
    np.random.seed(1)
    AL, Ds, caches = _propagate_forward(parameters, X, keep_prob=0.7)
    self.assertTrue(np.allclose(AL, np.array([[0.36974721, 0.00305176, 0.04565099, 0.49683389, 0.36974721]])))
    self.assertEqual(len(caches), 3)
  
  def test__step_backward(self):
    dAL = np.array([[-0.41675785, -0.05626683]])
    A_prev = np.array([
      [-2.1361961, 1.64027081],
      [-1.79343559, -0.84174737],
      [0.50288142, -1.24528809]
    ])
    W = np.array([[-1.05795222, -0.90900761, 0.55145404]])
    b =  np.array([[2.29220801]])
    Z = np.array([[0.04153939, -1.11792545]])
    
    cache = ((A_prev, W, b), (Z, 'sigmoid'))
    dA_prev, dW, db = _step_backward(dAL, 1, cache, None, None, 1)

    self.assertTrue(np.allclose(dA_prev, np.array([[0.11017994, 0.01105339], [0.09466817, 0.00949723], [-0.05743092, -0.00576154]])))
    self.assertTrue(np.allclose(dW, np.array([[0.10266786, 0.09778551, -0.01968084]])))
    self.assertTrue(np.allclose(db, [[-0.05729622]]))
    
    cache = ((A_prev, W, b), (Z, 'relu'))
    dA_prev, dW, db = _step_backward(dAL, 1, cache, None, None, 1)

    self.assertTrue(np.allclose(dA_prev, np.array([[0.44090989, 0.], [0.37883606, 0.], [-0.2298228, 0.]])))
    self.assertTrue(np.allclose(dW, np.array([[0.44513824, 0.37371418, -0.10478989]])))
    self.assertTrue(np.allclose(db, [[-0.20837892]]))

  def test__propagate_back(self):
    # AL = np.array([[1.78862847, 0.43650985]])
    # Y = np.array([[1, 0]])
    # A_prev = np.array([
    #   [0.09649747, -1.8634927],
    #   [-0.2773882 , -0.35475898],
    #   [-0.08274148, -0.62700068],
    #   [-0.04381817, -0.47721803]
    # ])
    # W = np.array([
    #   [-1.31386475,  0.88462238,  0.88131804,  1.70957306],
    #   [0.05003364, -0.40467741, -0.54535995, -1.54647732],
    #   [0.98236743, -1.10106763, -1.18504653, -0.2056499]
    # ])
    # b = np.array([
    #   [1.48614836],
    #   [0.23671627],
    #   [-1.02378514]
    # ])
    # Z = np.array([
    #   [-0.7129932 ,  0.62524497],
    #   [-0.16051336, -0.76883635],
    #   [-0.23003072,  0.74505627]
    # ])
    # first_cache = ((A_prev, W, b), (Z, 'relu'))

    # A_prev = np.array([
    #   [1.97611078, -1.24412333],
    #   [-0.62641691, -0.80376609],
    #   [-2.41908317, -0.92379202]
    # ])
    # W = np.array([[-1.02387576,  1.12397796, -0.13191423]])
    # b = np.array([[-1.62328545]])
    # Z = np.array([[0.64667545, -0.35627076]])
    
    # second_cache = ((A_prev, W, b), (Z, 'sigmoid'))
    
    # caches = [first_cache, second_cache]

    # expected = {
    #   'dW1': np.array([
    #     [ 0.41010002, 0.07807203, 0.13798444, 0.10502167],
    #     [0.,0.,0.,0.],
    #     [0.05283652, 0.01005865, 0.01777766, 0.0135308]
    #   ]),
    #   'db1': np.array([[-0.22007063], [0.], [-0.02835349]]),
    #   'dW2': np.array([[-0.39202432, -0.13325855, -0.04601089]]),
    #   'db2': np.array([[ 0.15187861]])
    # }

    # grads = _propagate_back(AL, Y, [1,1], caches, None, None, 1)

    # self.assertTrue(np.allclose(expected['dW1'], grads['dW1']))
    # self.assertTrue(np.allclose(expected['db1'], grads['db1']))
    # self.assertTrue(np.allclose(expected['dW2'], grads['dW2']))
    # self.assertTrue(np.allclose(expected['db2'], grads['db2']))

    # # regularization

    # X = np.array([
    #   [1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
    #   [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038],
    #   [1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944]
    # ])

    # AL = np.array([[ 0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]])

    # Y = np.array([[1, 1, 0, 1, 0]])

    # caches = (
    #   ((
    #     X,
    #     np.array([
    #       [-1.09989127, -0.17242821, -0.87785842],
    #       [0.04221375,  0.58281521, -1.10061918]
    #     ]),
    #     np.array([[1.14472371], [0.90159072]])
    #   ),
    #   (
    #     np.array([
    #       [-1.52855314, 3.32524635, 2.13994541, 2.60700654, -0.75942115],
    #       [-1.98043538,  4.1600994 ,  0.79051021,  1.46493512, -0.45506242]
    #     ]),
    #     "relu"
    #   )),
    #   ((
    #     np.array([
    #       [0., 3.32524635, 2.13994541, 2.60700654, 0.],
    #       [0., 4.1600994, 0.79051021, 1.46493512, 0.]
    #     ]),
    #     np.array([
    #       [0.50249434, 0.90085595],
    #       [-0.68372786, -0.12289023],
    #       [-0.93576943, -0.26788808]
    #     ]),
    #     np.array([
    #       [0.53035547],
    #       [-0.69166075],
    #       [-0.39675353]
    #     ])
    #   ),
    #   (
    #     np.array([
    #       [0.53035547, 5.94892323, 2.31780174, 3.16005701, 0.53035547],
    #       [-0.69166075, -3.47645987, -2.25194702, -2.65416996, -0.69166075],
    #       [-0.39675353, -4.62285846, -2.61101729, -3.22874921, -0.39675353]
    #     ]),
    #     "relu"
    #   )),
    #   ((
    #     np.array([
    #       [0.53035547, 5.94892323, 2.31780174, 3.16005701, 0.53035547],
    #       [0., 0., 0., 0., 0.],
    #       [0., 0., 0., 0., 0.]
    #     ]),
    #     np.array([[-0.6871727, -0.84520564, -0.67124613]]),
    #     np.array([[-0.0126646]])
    #   ),
    #   (
    #     np.array([[-0.3771104, -4.10060224, -1.60539468, -2.18416951, -0.3771104 ]]),
    #     "sigmoid"
    #   ))
    # )

    # expected = {
    #   'dW1': np.array([
    #     [-0.25604646, 0.12298827, -0.28297129],
    #     [-0.17706303, 0.34536094, -0.4410571 ]
    #   ]),
    #   'db1': np.array([[0.11845855], [0.21236874]]),
    #   'dW2': np.array([
    #     [0.79276486, 0.85133918],
    #     [-0.0957219, -0.01720463],
    #     [-0.13100772, -0.03750433]
    #   ]),
    #   'db2': np.array([[ 0.26135226], [ 0.], [ 0.]]),
    #   'dW3': np.array([[-1.77691347, -0.11832879, -0.09397446]]),
    #   'db3': np.array([[-0.38032981]])
    # }
    
    # grads = _propagate_back(AL, Y, [1, 1, 1], caches, "L2", 0.7, 1)
    
    # self.assertTrue(np.allclose(grads['dW1'], expected['dW1']))
    # self.assertTrue(np.allclose(grads['db1'], expected['db1']))
    # self.assertTrue(np.allclose(grads['dW2'], expected['dW2']))
    # self.assertTrue(np.allclose(grads['db2'], expected['db2']))
    # self.assertTrue(np.allclose(grads['dW3'], expected['dW3']))
    # self.assertTrue(np.allclose(grads['db3'], expected['db3']))

    # Dropout

    AL = np.array([[0.32266394, 0.49683389, 0.00348883, 0.49683389, 0.32266394]])

    Y = np.array([[1, 1, 0, 1, 0]])

    caches = (
      ((
        np.array([
          [1.62434536, -0.61175641, -0.52817175, -1.07296862, 0.86540763],
          [-2.3015387, 1.74481176, -0.7612069, 0.3190391, -0.24937038],
          [ 1.46210794, -2.06014071, -0.3224172, -0.38405435, 1.13376944]
        ]),
        np.array([[-1.09989127, -0.17242821, -0.87785842], [0.04221375, 0.58281521, -1.10061918]]),
        np.array([[1.14472371], [0.90159072]])
      ),
      (
        np.array([
          [-1.52855314, 3.32524635, 2.13994541, 2.60700654, -0.75942115],
          [-1.98043538, 4.1600994, 0.79051021, 1.46493512, -0.45506242]
        ]),
        "relu"
      )),
      ((
        np.array([
          [0., 0., 4.27989081, 5.21401307, 0.],
          [0., 8.32019881, 1.58102041, 2.92987024, 0.]
        ]),
        np.array([
          [0.50249434, 0.90085595],
          [-0.68372786, -0.12289023],
          [-0.93576943, -0.26788808]
        ]),
        np.array([[0.53035547], [-0.69166075], [-0.39675353]])
      ),
      (
        np.array([
          [0.53035547, 8.02565606, 4.10524802, 5.78975856, 0.53035547],
          [-0.69166075, -1.71413186, -3.81223329, -4.61667916, -0.69166075],
          [-0.39675353, -2.62563561, -4.82528105, -6.0607449, -0.39675353]
        ]),
        "relu"
      )),
      ((
        np.array([
          [1.06071093, 0., 8.21049603, 0., 1.06071093],
          [0., 0., 0., 0., 0.],
          [0., 0., 0., 0., 0.]
        ]),
        np.array([[-0.6871727, -0.84520564, -0.67124613]]),
        np.array([[-0.0126646]])
      ),
      (
        np.array([[-0.7415562, -0.0126646, -5.65469333, -0.0126646, -0.7415562 ]]),
        "sigmoid"
      ))
    )

    expected = {
      'dW1': np.array([[0.00019884, 0.00028657, 0.00012138], [ 0.00035647, 0.00051375, 0.00021761]]),
      'db1': np.array([[-0.00037647], [-0.00067492]]),
      'dW2': np.array([[-0.00256518, -0.0009476 ], [0., 0.], [0., 0.]]),
      'db2': np.array([[ 0.06033089], [0.], [0.]]),
      'dW3': np.array([[-0.06951191, 0., 0.]]),
      'db3': np.array([[-0.2715031]])
    }

    Ds =  [
            1,
            np.array([[1, 0, 1, 1, 1], [1, 1, 1, 1, 0]]),
            np.array([[1, 0, 1, 0, 1], [0, 1, 0, 1, 1], [0, 0, 1, 0, 0]])
          ]

    grads = _propagate_back(AL, Y, Ds, caches, None, None, 0.8)
    self.assertTrue(np.allclose(grads['dW1'], expected['dW1']))
    self.assertTrue(np.allclose(grads['db1'], expected['db1']))
    self.assertTrue(np.allclose(grads['dW2'], expected['dW2']))
    self.assertTrue(np.allclose(grads['db2'], expected['db2']))
    self.assertTrue(np.allclose(grads['dW3'], expected['dW3']))
    self.assertTrue(np.allclose(grads['db3'], expected['db3']))

  def test__cost(self):
    Y, A = np.array([[1, 1, 1]]), np.array([[0.8, 0.9, 0.4]])
    self.assertEqual(_cost(A, Y, None, None, None), 0.414931599615397)

    Y, A =  np.array([[1, 1, 0, 1, 0]]), np.array([[0.40682402, 0.01629284, 0.16722898, 0.10118111, 0.40682402]])
    parameters = {
      'W1': np.array([
        [1.62434536, -0.61175641, -0.52817175],
        [-1.07296862, 0.86540763, -2.3015387]
      ]),
      'b1': np.array([
        [1.74481176],
        [-0.7612069 ]
      ]),
      'W2': np.array([
        [0.3190391, -0.24937038],
        [1.46210794, -2.06014071],
        [-0.3224172, -0.38405435]
      ]),
      'b2': np.array([
        [1.13376944],
        [-1.09989127],
        [-0.17242821]
      ]),
      'W3': np.array([[-0.87785842,  0.04221375,  0.58281521]]),
      'b3': np.array([[-1.10061918]])
    }

    self.assertEqual(_cost(A, Y, parameters, regularization="L2", lambd=0.1), 1.786485945169561)
  
  def test_propagate(self):
    w, b, X, Y = np.array([[1., 2.]]), 2., np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([[1,0,1]])
    parameters = {
      'W1': w,
      'b1': b
    }

    grads, cost = propagate(parameters, X, Y)
    self.assertTrue(np.allclose(grads['dW1'], np.array([[0.99845601, 2.39507239]])))
    self.assertTrue(np.allclose(grads['db1'], np.array([[0.00145558]])))
    self.assertEqual(cost, 5.801545319394553)

  def test__tf_propagate_forward(self):
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_tf_parameters([12288, 25, 12, 6], seed=1)
    tf.get_variable_scope().reuse_variables()
    Z = _tf_propagate_forward(parameters, X)
    np.random.seed(1)
    dict_X = np.random.randn(12288, 1080)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      Z = sess.run(Z, feed_dict={X: dict_X})

    self.assertAlmostEqual(Z[0][0], -2.46408725)
    self.assertAlmostEqual(Z[5][1079], -0.9831996)
    self.assertEqual(Z.shape, (6, 1080))

  def test__tf_cost(self):
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_tf_parameters([12288, 25, 12, 6], seed=1)
    tf.get_variable_scope().reuse_variables()
    Z = _tf_propagate_forward(parameters, X)
    cost = _tf_cost(Z, Y)
    np.random.seed(1)
    dict_X = np.random.randn(12288, 1080)
    dict_Y = np.random.randn(6, 1080)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
      sess.run(init)
      cost = sess.run(cost, feed_dict={X: dict_X, Y: dict_Y})

    self.assertAlmostEqual(cost, 0.11581959)

class TestGradientChecking(unittest.TestCase):

  def parameters(self):  
    return {
      'W1': np.array([
        [-0.3224172, -0.38405435, 1.13376944, -1.09989127],
        [-0.17242821, -0.87785842, 0.04221375, 0.58281521],
        [-1.10061918, 1.14472371, 0.90159072, 0.50249434],
        [ 0.90085595, -0.68372786, -0.12289023, -0.93576943],
        [-0.26788808, 0.53035547, -0.69166075, -0.39675353]
      ]),
      'b1': np.array([
        [-0.6871727],
        [-0.84520564],
        [-0.67124613],
        [-0.0126646],
        [-1.11731035]
      ]),
      'W2': np.array([
        [ 0.2344157, 1.65980218, 0.74204416, -0.19183555, -0.88762896],
        [-0.74715829, 1.6924546, 0.05080775, -0.63699565, 0.19091548],
        [ 2.10025514, 0.12015895, 0.61720311, 0.30017032, -0.35224985]
      ]),
      'b2': np.array([[-1.1425182], [-0.34934272], [-0.20889423]]),
      'W3': np.array([[ 0.58662319, 0.83898341, 0.93110208]]),
      'b3': np.array([[ 0.28558733]])
    }

  def gradients(self):
    return { 
      'dW1': np.array([
        [-0.37347779, -1.47903216, 0.17596143, -1.33685036],
        [-0.01967514, -0.08573553, 0.01188465, -0.07674312],
        [0.03916037, -0.05539735, 0.04872715, -0.09359393],
        [-0.05337778, -0.21138458, 0.02514856, -0.19106384],
        [0., 0., 0., 0.]
      ]),
      'db1': np.array([
        [0.63290787],
        [0.0372514 ],
        [-0.06401301],
        [0.09045575],
        [0.]
      ]),
      'dW2': np.array([
        [0., 0., 0., 0., 0.],
        [0., 0., 0., 0., 0.],
        [0.91580165, 0.02451548, -0.10797954, 0.90281891, 0.]
      ]),
      'db2': np.array([[0.], [0.], [0.19763343]]),
      'dW3': np.array([[0.,  0.,  2.24404238]]),
      'db3': np.array([[0.21225753]]),
    }
  
  def test__dictionary_to_vector_on_parameters(self):
    expected = np.array([
      [-0.3224172], [-0.38405435], [1.13376944], [-1.09989127], [-0.17242821], [-0.87785842],
      [0.04221375], [0.58281521], [-1.10061918], [1.14472371], [0.90159072], [0.50249434],
      [0.90085595], [-0.68372786], [-0.12289023], [-0.93576943], [-0.26788808], [0.53035547],
      [-0.69166075], [-0.39675353], [-0.6871727 ], [-0.84520564], [-0.67124613], [-0.0126646],
      [-1.11731035], [ 0.2344157], [1.65980218], [0.74204416], [-0.19183555], [-0.88762896],
      [-0.74715829], [1.6924546], [0.05080775], [-0.63699565], [0.19091548], [2.10025514],
      [0.12015895], [0.61720311], [0.30017032], [-0.35224985], [-1.1425182], [-0.34934272],
      [-0.20889423], [0.58662319], [0.83898341], [0.93110208], [0.28558733]
    ])

    parameter_values = _dictionary_to_vector(self.parameters())

    self.assertTrue(np.allclose(parameter_values, expected))

  
  def test__dictionary_to_vector_on_gradients(self):
    expected = np.array([
      [-0.37347779],
      [-1.47903216],
      [ 0.17596143],
      [-1.33685036],
      [-0.01967514],
      [-0.08573553],
      [0.01188465],
      [-0.07674312],
      [0.03916037],
      [-0.05539735],
      [0.04872715],
      [-0.09359393],
      [-0.05337778],
      [-0.21138458],
      [0.02514856],
      [-0.19106384],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.63290787],
      [0.0372514 ],
      [-0.06401301],
      [0.09045575],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.],
      [0.91580165],
      [0.02451548],
      [-0.10797954],
      [0.90281891],
      [0.],
      [0.],
      [0.],
      [0.19763343],
      [0.],
      [0.],
      [2.24404238],
      [0.21225753]
    ])

    gradient_values = _dictionary_to_vector(self.gradients(), key_bases=["dW", "db"])

    self.assertTrue(np.allclose(gradient_values, expected))

  
  def test_check_gradient(self):
    X = np.array([
      [1.62434536, -0.61175641, -0.52817175],
      [-1.07296862, 0.86540763, -2.3015387],
      [ 1.74481176, -0.7612069, 0.3190391],
      [-0.24937038, 1.46210794, -2.06014071]
    ])

    Y = np.array([[1, 1, 0]])

    difference = check_gradient(self.parameters(), self.gradients(), X, Y)
    self.assertEqual(difference, 7.058343235500508e-08)

class TestTF_Helpers(unittest.TestCase):
  def test_create_placeholders(self):
    X, Y = create_placeholders(12288, 6)
    print ("X = " + str(X))
    print ("Y = " + str(Y))

    self.assertEqual(X.shape[0], 12288)
    self.assertEqual(Y.shape[0], 6)

    self.assertEqual(X.dtype, tf.float32)
    self.assertEqual(Y.dtype, tf.float32)

class TestConvolutions(unittest.TestCase):
  def test_zero_pad(self):
    np.random.seed(1)
    x = np.random.randn(4, 3, 3, 2)
    x_pad = zero_pad(x, 2)

    oneone = np.array([[0.90085595, -0.68372786], [-0.12289023, -0.93576943], [-0.26788808, 0.53035547]])
    padone = np.zeros((7,2))

    self.assertEqual(x.shape, (4,3,3,2))
    self.assertEqual(x_pad.shape, (4,7,7,2))
    self.assertTrue(np.allclose(x[1,1], oneone))
    self.assertTrue(np.allclose(x_pad[1,1], padone))

  def test_conv_step(self):
    np.random.seed(1)
    a_slice_prev = np.random.randn(4, 4, 3)
    W = np.random.randn(4, 4, 3)
    b = np.random.randn(1, 1, 1)

    Z = conv_step(a_slice_prev, W, b)

    self.assertAlmostEqual(Z, -6.99908945068)

  def test_conv_forward(self):
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = { "pad" : 2, "stride": 2 }

    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    zthreetwoone = np.array([-0.61490741, -6.7439236, -2.55153897, 1.75698377, 3.56208902, 0.53036437, 5.18531798, 8.75898442])
    cczeroonetwothree = np.array([-0.20075807, 0.18656139, 0.41005165])

    self.assertAlmostEqual(np.mean(Z), 0.0489952035289)
    self.assertTrue(np.allclose(Z[3,2,1], zthreetwoone))
    self.assertTrue(np.allclose(cache_conv[0][1][2][3], cczeroonetwothree))

  def test_pool_forward(self):
    np.random.seed(1)
    A_prev = np.random.randn(2, 4, 4, 3)
    hparameters = { "stride" : 2, "f": 3 }

    A, cache = pool_forward(A_prev, hparameters)

    eA = np.array([[[[1.74481176, 0.86540763, 1.13376944]]], [[[1.13162939, 1.51981682, 2.18557541]]]])
    self.assertTrue(np.allclose(A, eA))

    A, cache = pool_forward(A_prev, hparameters, mode = "average")
    eA = np.array([[[[0.02105773, -0.20328806, -0.40389855]]], [[[-0.22154621, 0.51716526, 0.48155844]]]])

  def test_conv_backward(self):
    np.random.seed(1)
    A_prev = np.random.randn(10,4,4,3)
    W = np.random.randn(2,2,3,8)
    b = np.random.randn(1,1,1,8)
    hparameters = { "pad" : 2, "stride": 2 }
    Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
    
    np.random.seed(1)
    dA, dW, db = conv_backward(Z, cache_conv)

    self.assertAlmostEqual(np.mean(dA), 1.45243777754)
    self.assertAlmostEqual(np.mean(dW), 1.72699145831)
    self.assertAlmostEqual(np.mean(db), 7.83923256462)

  def test_create_mask_from_window(self):
    np.random.seed(1)
    x = np.random.randn(2,3)
    mask = create_mask_from_window(x)

    expected = np.array([[True, False, False], [False, False, False]])
    self.assertTrue(np.array_equal(mask, expected))

  def test_distribute_value(self):
    a = distribute_value(2, (2,2))
    expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    self.assertTrue(np.array_equal(a, expected))

  def test_pool_backward(self):
    np.random.seed(1)
    A_prev = np.random.randn(5, 5, 3, 2)
    hparameters = { "stride" : 1, "f": 2 }
    A, cache = pool_forward(A_prev, hparameters)
    dA = np.random.randn(5, 4, 2, 2)

    dA_prev = pool_backward(dA, cache, mode = "max")
    oneone = np.array([[0., 0.], [5.05844394, -1.68282702], [0., 0.]])

    self.assertTrue(np.allclose(dA_prev[1,1], oneone))

    dA_prev = pool_backward(dA, cache, mode = "average")
    oneone = np.array([[0.08485462, 0.2787552], [1.26461098, -0.25749373], [1.17975636, -0.53624893]])

    self.assertTrue(np.allclose(dA_prev[1,1], oneone))

  def test_initialize_tf_parameters(self):
    parameters = initialize_tf_conv_parameters([[4, 4, 3, 8], [2, 2, 8, 16]], seed=1)
    init = tf.global_variables_initializer()
    with tf.Session() as sess_test:
      sess_test.run(init)
      parameters = sess_test.run(parameters)

    expected = {
      'W1': np.array([0.00131723, 0.14176141, -0.04434952, 0.09197326, 0.14984085, -0.03514394, -0.06847463, 0.05245192]),
      'W2': np.array([-0.08566415, 0.17750949, 0.11974221, 0.16773748, -0.0830943, -0.08058, -0.00577033, -0.14643836, 0.24162132, -0.05857408, -0.19055021, 0.1345228, -0.22779644, -0.1601823, -0.16117483, -0.10286498])
    }

    self.assertTrue(np.allclose(parameters['W1'][1,1,1], expected['W1']))
    self.assertTrue(np.allclose(parameters['W2'][1,1,1], expected['W2']))
    



if __name__ == '__main__':
  unittest.main()