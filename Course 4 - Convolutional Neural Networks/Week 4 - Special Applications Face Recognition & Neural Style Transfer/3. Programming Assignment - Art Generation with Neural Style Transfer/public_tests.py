import tensorflow as tf
from tensorflow.python.framework.ops import EagerTensor
import numpy as np

### ex 1
def compute_content_cost_test(target):
    tf.random.set_seed(1)
    a_C = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random.normal([1, 1, 4, 4, 3], mean=1, stddev=4)
    J_content = target(a_C, a_G)
    J_content_0 = target(a_C, a_C)
    assert type(J_content) == EagerTensor, "Use the tensorflow function"
    assert np.isclose(J_content_0, 0.0), "Wrong value. compute_content_cost(A, A) must be 0"
    assert np.isclose(J_content, 7.0568767), f"Wrong value. Expected {7.0568767},  current{J_content}"

    print("J_content = " + str(J_content))

    # Test that it works with symbolic tensors
    ll = tf.keras.layers.Dense(8, activation='relu', input_shape=(1, 4, 4, 3))
    model_tmp = tf.keras.models.Sequential()
    model_tmp.add(ll)
    try:
        target(ll.output, ll.output)
        print("\033[92mAll tests passed")
    except Exception as inst:
        print("\n\033[91mDon't use the numpy API inside compute_content_cost\n")
        print(inst)
        
        
### ex 2
def gram_matrix_test(target):
    tf.random.set_seed(1)
    A = tf.random.normal([3, 2 * 1], mean=1, stddev=4)
    GA = target(A)

    assert type(GA) == EagerTensor, "Use the tensorflow function"
    assert GA.shape == (3, 3), "Wrong shape. Check the order of the matmul parameters"
    assert np.allclose(GA[0,:], [63.193256, -26.729713, -7.732155]), "Wrong values."

    print("GA = \n" + str(GA))

    print("\033[92mAll tests passed")
    
    
### ex 3
def compute_layer_style_cost_test(target):
    tf.random.set_seed(1)
    a_S = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
    a_G = tf.random.normal([1, 4, 4, 3], mean=1, stddev=4)
    J_style_layer_GG = target(a_G, a_G)
    J_style_layer_SG = target(a_S, a_G)


    assert type(J_style_layer_GG) == EagerTensor, "Use the tensorflow functions"
    assert np.isclose(J_style_layer_GG, 0.0), "Wrong value. compute_layer_style_cost(A, A) must be 0"
    assert J_style_layer_SG > 0, "Wrong value. compute_layer_style_cost(A, B) must be greater than 0 if A != B"
    assert np.isclose(J_style_layer_SG, 14.01649), "Wrong value."

    print("J_style_layer = " + str(J_style_layer_SG))
    print("\033[92mAll tests passed")
    
### ex 4 is already implemented for the learners


### ex 5
def total_cost_test(target):
    J_content = 0.2    
    J_style = 0.8
    J = target(J_content, J_style)

    assert type(J) == EagerTensor, "Do not remove the @tf.function() modifier from the function"
    assert J == 34, "Wrong value. Try inverting the order of alpha and beta in the J calculation"
    assert np.isclose(target(0.3, 0.5, 3, 8), 4.9), "Wrong value. Use the alpha and beta parameters"

    np.random.seed(1)
    print("J = " + str(target(np.random.uniform(0, 1), np.random.uniform(0, 1))))

    print("\033[92mAll tests passed")
    

### ex 6
def train_step_test(target, generated_image):
    generated_image = tf.Variable(generated_image)


    J1 = target(generated_image)
    print(J1)
    assert type(J1) == EagerTensor, f"Wrong type {type(J1)} != {EagerTensor}"
    assert np.isclose(J1, 25629.055, rtol=0.05), f"Unexpected cost for epoch 0: {J1} != {25629.055}"

    J2 = target(generated_image)
    print(J2)
    assert np.isclose(J2, 17812.627, rtol=0.05), f"Unexpected cost for epoch 1: {J2} != {17735.512}"

    print("\033[92mAll tests passed")
    
    
    
    
    
    