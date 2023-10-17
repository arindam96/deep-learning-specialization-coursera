import numpy as np
from numpy import array
from testCases import *
from dlai_tools.testing_utils import single_test, multiple_test

### ex 1         
def update_parameters_with_gd_test(target):
    parameters, grads, learning_rate = update_parameters_with_gd_test_case()
    learning_rate = 0.01
    
    expected_output = {'W1': np.array([[ 1.63312395, -0.61217855, -0.5339999],
                                       [-1.06196243,  0.85396039, -2.3105546]]),
                       'b1': np.array([[ 1.73978682],
                                       [-0.77021546]]),
                       'W2': np.array([[ 0.32587637, -0.24814147],
                                       [ 1.47146563, -2.05746183],
                                       [-0.32772076, -0.37713775]]),
                       'b2': np.array([[ 1.13773698],
                                       [-1.09301954],
                                       [-0.16397615]])}

    params_up = target(parameters, grads, learning_rate)

    for key in params_up.keys():
        assert type(params_up[key]) == np.ndarray, f"Wrong type for {key}. We expected np.ndarray, but got {type(params_up[key])}"
        assert params_up[key].shape == parameters[key].shape, f"Wrong shape for {key}. {params_up[key].shape} != {parameters[key].shape}"
        assert np.allclose(params_up[key], expected_output[key]), f"Wrong values for {key}. Check the formulas. Expected: \n {expected_output[key]}"
    
    print("\033[92mAll tests passed")
            
### ex 2        
def random_mini_batches_test(target):
    np.random.seed(1)
    mini_batch_size = 2
    X = np.random.randn(5, 7)
    Y = np.random.randn(1, 7) < 0.5

    expected_output = [(np.array([[ 1.74481176, -0.52817175],
                                  [-0.38405435, -0.24937038],
                                  [-1.10061918, -0.17242821],
                                  [-0.93576943,  0.50249434],
                                  [-0.67124613, -0.69166075]]), 
                        np.array([[ True,  True]])), 
                       (np.array([[-0.61175641, -1.07296862],
                                  [ 0.3190391 ,  1.46210794],
                                  [-1.09989127, -0.87785842],
                                  [ 0.90159072,  0.90085595],
                                  [ 0.53035547, -0.39675353]]), 
                        np.array([[ True, False]])), 
                       (np.array([[ 1.62434536, -2.3015387 ],
                                  [-0.7612069 , -0.3224172 ],
                                  [ 1.13376944,  0.58281521],
                                  [ 1.14472371, -0.12289023],
                                  [-0.26788808, -0.84520564]]), 
                        np.array([[ True,  True]])), 
                       (np.array([[ 0.86540763],
                                  [-2.06014071],
                                  [ 0.04221375],
                                  [-0.68372786],
                                  [-0.6871727 ]]), 
                        np.array([[False]]))]
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [X, Y, mini_batch_size],
            "expected": expected_output,
            "error": "Wrong output"
        }
    ]
    
    multiple_test(test_cases, target)

    
### ex 3    
def initialize_velocity_test(target):
    parameters = initialize_velocity_test_case()
    
    expected_output = {'dW1': np.array([[0., 0.],
                                        [0., 0.],
                                        [0., 0.]]), 
                       'db1': np.array([[0.],
                                        [0.],
                                        [0.]]), 
                       'dW2': np.array([[0., 0., 0.],
                                        [0., 0., 0.],
                                        [0., 0., 0.]]), 
                       'db2': array([[0.],
                                     [0.],
                                     [0.]])}
    
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)

### ex 4
def update_parameters_with_momentum_test(target):
    parameters, grads, v = update_parameters_with_momentum_test_case()
    beta = 0.9
    learning_rate = 0.01
    
    expected_parameters = {'W1': np.array([[ 1.62522322, -0.61179863, -0.52875457],
                                           [-1.071868,    0.86426291, -2.30244029]]),
                           'b1': np.array([[ 1.74430927],
                                           [-0.76210776]]),
                           'W2': np.array([[ 0.31972282, -0.24924749],
                                           [ 1.46304371, -2.05987282],
                                           [-0.32294756, -0.38336269]]),
                           'b2': np.array([[ 1.1341662 ],
                                           [-1.09920409],
                                           [-0.171583  ]])}
    
    expected_v = {'dW1': np.array([[-0.08778584,  0.00422137,  0.05828152],
                                   [-0.11006192,  0.11447237,  0.09015907]]),
                  'dW2': np.array([[-0.06837279, -0.01228902],
                                   [-0.09357694, -0.02678881],
                                   [ 0.05303555, -0.06916608]]),
                  'db1': np.array([[0.05024943],
                                   [0.09008559]]),
                  'db2': np.array([[-0.03967535],
                                   [-0.06871727],
                                   [-0.08452056]])}
    
    expected_output = (expected_parameters, expected_v)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error":"Datatype mismatch"
        },
        {
            "name": "shape_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters, grads, v, beta, learning_rate],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)    

    
### ex 5   
def initialize_adam_test(target):
    parameters = initialize_adam_test_case()
    
    expected_v = {'dW1': np.array([[0., 0., 0.],
                                   [0., 0., 0.]]),
                  'db1': np.array([[0.],
                                   [0.]]),
                  'dW2': np.array([[0., 0.],
                                   [0., 0.],
                                   [0., 0.]]),
                  'db2': np.array([[0.],
                                   [0.],
                                   [0.]])}
    
    expected_s = {'dW1': np.array([[0., 0., 0.],
                                   [0., 0., 0.]]),
                  'db1': np.array([[0.],
                                   [0.]]),
                  'dW2': np.array([[0., 0.],
                                   [0., 0.],
                                   [0., 0.]]),
                  'db2': np.array([[0.],
                                   [0.],
                                   [0.]])}
    
    expected_output = (expected_v, expected_s)
    test_cases = [
        {
            "name":"datatype_check",
            "input": [parameters],
            "expected": expected_output,
            "error":"The function should return a numpy array."
        },
        {
            "name": "shape_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong shape"
        },
        {
            "name": "equation_output_check",
            "input": [parameters],
            "expected": expected_output,
            "error": "Wrong output"
        } 
    ]
    
    multiple_test(test_cases, target)
    
### ex 6    
def update_parameters_with_adam_test(target):
    parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon = update_parameters_with_adam_test_case()

    c1 = 1.0 / (1 - beta1**t)
    c2 = 1.0 / (1 - beta2**t)
    
    expected_v = {'dW1': np.array([-0.17557168,  0.00844275,  0.11656304]), 
                  'dW2': np.array([-0.13674557, -0.02457805]), 
                  'db1': np.array([0.10049887]), 
                  'db2': np.array([-0.07935071])}
    
    expected_s = {'dW1': np.array([0.08631117, 0.00019958, 0.03804344]),
                  'dW2':np.array([0.05235818, 0.00169142]),
                  'db1':np.array([0.02828006]),
                  'db2':np.array([0.0176303 ])}
    
    expected_parameters = {'W1': np.array([ 1.63937725, -0.62327448, -0.54308727]),
                           'W2':np.array([ 0.33400549, -0.23563857]),
                           'b1':np.array([ 1.72995096]),
                           'b2':np.array([ 1.14852557])}

    parameters, v, s, vc, sc  = target(parametersi, grads, vi, si, t, learning_rate, beta1, beta2, epsilon)
    
    for key in v.keys():
        
        assert type(v[key]) == np.ndarray, f"Wrong type for v['{key}']. Expected np.ndarray"
        assert v[key].shape == vi[key].shape, f"Wrong shape for  v['{key}']. The update must keep the dimensions of v inputs"
        assert np.allclose(v[key][0], expected_v[key]), f"Wrong values. Check you formulas for v['{key}']"
        #print(f"v[\"{key}\"]: \n {str(v[key][0])}")

    for key in vc.keys():
        assert type(vc[key]) == np.ndarray, f"Wrong type for v_corrected['{key}']. Expected np.ndarray"
        assert vc[key].shape == vi[key].shape, f"Wrong shape for  v_corrected['{key}']. The update must keep the dimensions of v inputs"
        assert np.allclose(vc[key][0], expected_v[key] * c1), f"Wrong values. Check you formulas for v_corrected['{key}']"
        #print(f"vc[\"{key}\"]: \n {str(vc[key])}")

    for key in s.keys():
        assert type(s[key]) == np.ndarray, f"Wrong type for s['{key}']. Expected np.ndarray"
        assert s[key].shape == si[key].shape, f"Wrong shape for  s['{key}']. The update must keep the dimensions of s inputs"
        assert np.allclose(s[key][0], expected_s[key]), f"Wrong values. Check you formulas for s['{key}']"
        #print(f"s[\"{key}\"]: \n {str(s[key])}")

    for key in sc.keys():
        assert type(sc[key]) == np.ndarray, f"Wrong type for s_corrected['{key}']. Expected np.ndarray"
        assert sc[key].shape == si[key].shape, f"Wrong shape for  s_corrected['{key}']. The update must keep the dimensions of s inputs"
        assert np.allclose(sc[key][0], expected_s[key] * c2), f"Wrong values. Check you formulas for s_corrected['{key}']"   
        # print(f"sc[\"{key}\"]: \n {str(sc[key])}")

    for key in parameters.keys():
        assert type(parameters[key]) == np.ndarray, f"Wrong type for parameters['{key}']. Expected np.ndarray"
        assert parameters[key].shape == parametersi[key].shape, f"Wrong shape for  parameters['{key}']. The update must keep the dimensions of parameters inputs"
        assert np.allclose(parameters[key][0], expected_parameters[key]), f"Wrong values. Check you formulas for parameters['{key}']"   
        #print(f"{key}: \n {str(parameters[key])}")

    print("\033[92mAll tests passed")


### ex 7    
def update_lr_test(target):
    learning_rate = 0.5
    epoch_num = 2
    decay_rate = 1
    expected_output = 0.16666666666666666
    
    output = target(learning_rate, epoch_num, decay_rate)
    
    assert np.isclose(output, expected_output), f"output: {output} expected: {expected_output}"
    print("\033[92mAll tests passed")


### ex 8    
def schedule_lr_decay_test(target):
    learning_rate = 0.5
    epoch_num_1 = 100
    epoch_num_2 = 10
    decay_rate = 1
    time_interval = 100
    expected_output_1 = 0.25
    expected_output_2 = 0.5
    
    output_1 = target(learning_rate, epoch_num_1, decay_rate, time_interval)
    output_2 = target(learning_rate, epoch_num_2, decay_rate, time_interval)


    assert np.isclose(output_1, expected_output_1),f"output: {output_1} expected: {expected_output_1}"
    assert np.isclose(output_2, expected_output_2),f"output: {output_2} expected: {expected_output_2}"
    
    learning_rate = 0.3
    epoch_num_1 = 1000
    epoch_num_2 = 100
    decay_rate = 0.25
    time_interval = 100
    expected_output_1 = 0.085714285
    expected_output_2 = 0.24

    output_1 = target(learning_rate, epoch_num_1, decay_rate, time_interval)
    output_2 = target(learning_rate, epoch_num_2, decay_rate, time_interval)


    assert np.isclose(output_1, expected_output_1),f"output: {output_1} expected: {expected_output_1}"
    assert np.isclose(output_2, expected_output_2),f"output: {output_2} expected: {expected_output_2}"

    print("\033[92mAll tests passed")
    