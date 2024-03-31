WeChat: cstutorcs
QQ: 749389476
Email: tutorcs@163.com
import numpy as np
import re

def sigmoid(evidence):
    """Applies the logistic function to convert evidence to probability."""
    return 1 / (1 + np.exp(-evidence))

def sigmoid_derivative(logistic_output):
    """Returns the derivative of the logistic function, calculated from the
    output of the logistic function at a particular point.
    """
    return logistic_output * (1 - logistic_output)

def tanh(z):
    """Applies the tanh function to calculate activation, given input z."""
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


def tanh_derivative(tanh_output):
    """Returns the derivative of the tanh function, calculated from the
    output of the tanh function at a particular point.
    """
    return 1 - (tanh_output ** 2)


def relu(z):
    """Applies the ReLU function to calculate activation, given input z."""
    return np.maximum(z, 0)


def relu_derivative(relu_output):
    """Returns the derivative of the ReLU function, calculated from the
    output of the ReLU function at a particular point.
    """
    if isinstance(relu_output, np.ndarray):
        return (relu_output > 0).astype(int)
    else:
        return int(relu_output > 0)
    

def tidy_print(x):
    return (x.__name__ if callable(x) else re.sub(r"\s{2,}", " ", repr(x).replace("array", "np.array")).replace(" ,", ",").replace("[ ", "["))
    

def check_function(function, expected, message, *args, **kwargs):
    """Checks a function by providing it with *args and **kwargs and ensuring
    that its output matches what's expected. If not, prints the provided message.
    Returns a boolean indicating whether the test passed."""
    result = function(*args, **kwargs)
    passed = equal(result, expected)
    if not passed:
        print(message)
        print("\nFor input: {}({})".format(function.__name__, ", ".join([tidy_print(a) for a in args]) + (", " if kwargs else "") + ", ".join(["{}={}".format(k, tidy_print(v)) for (k, v) in kwargs.items()])))
        print("Expected output: {}".format(tidy_print(expected)))
        print("Got: {}".format(tidy_print(result)))
    return passed


def equal(obj1, obj2):
    """Checks if two objects are equal"""
    # Check they have equivalent types
    if isinstance(obj1, (int, float)):
        e = isinstance(obj2, (int, float))
    else:
        e = isinstance(obj1, type(obj2))
    
    # Check they have equivalent values
    if isinstance(obj1, np.ndarray):
        e = e and obj1.shape == obj2.shape and np.allclose(obj1, obj2)
    elif isinstance(obj1, (list, tuple)):
        e = e and len(obj1) == len(obj2)
        for (elem1, elem2) in zip(obj1, obj2):
            if not e:
                break
            e = e and equal(elem1, elem2)
    else:
        e = e and obj1 == obj2
        
    return e


def check(function):
    """Wrapper for convenient use of check_function"""
    function_name = function.__name__
    
    expected_list = list()
    message_list = list()
    args_list = list()
    kwargs_list = list()
    
    if function_name == "recurrent_linear":
        current_values = np.array([1, 2, 3])
        input_weights = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0]])
        prev_hidden = np.array([2, 1])
        recurrent_weights = np.array([[0, 1], [-1, 1]])

        args_list = [(current_values, input_weights),
                     (current_values, input_weights)]
        kwargs_list = [{"prev_hidden": prev_hidden, "recurrent_weights": recurrent_weights},
                       {}]
        expected_list = [np.array([1, 0]),
                         np.array([0, 1])]
        message_list = ["Function does not incorporate previous hidden layer",
                        "Function tries to incorporate previous hidden layer when there isn't one"]
    
    elif function_name == "get_hidden":
        input_weights = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0]])
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        values_1 = np.array([1, 2, 3], dtype="float64")
        values_2 = np.array([0, 0, 1], dtype="float64")
        hidden_1 = np.array([0, 1], dtype="float64")
        
        args_list = [(hidden_1, recurrent_weights, values_2, input_weights),
                     (None, recurrent_weights, values_1, input_weights)]
        kwargs_list = [{},
                       {"activation_function": sigmoid}]
        expected_list = [np.array([0, 0]),
                         np.array([0.5, 0.73105858])]
        message_list = ["Check you are using recurrent_linear() and activation_function()",
                        "Function does not use the activation_function provided as an argument"]
    
    elif function_name == "get_hidden_sequence":
        input_weights = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0]])
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        input_values = [np.array([1, 2, 3], dtype="float64"),
                        np.array([0, 0, 1], dtype="float64")]
        
        args_list = [(input_values, input_weights, recurrent_weights),
                     (input_values, input_weights, recurrent_weights)]
        kwargs_list = [{},
                       {"activation_function": sigmoid}]
        expected_list = [[np.array([0., 1.]), np.array([0., 0.])],
                         [np.array([0.5, 0.73105858]), np.array([0.43316699, 0.31670814])]]
        message_list = ["Incorrect",
                        "Function does not use the activation_function provided as an argument"]

    elif function_name == "recurrent_classifier_forward":
        input_weights = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0]])
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        output_weights = np.array([1, 1, -1])
        input_values = [np.array([1, 2, 3], dtype="float64"),
                        np.array([0, 1, 1], dtype="float64")]
        
        args_list = [(input_values, input_weights, recurrent_weights, output_weights),
                     (input_values, input_weights, recurrent_weights, output_weights)]
        kwargs_list = [{},
                       {"activation_function": sigmoid}]
        expected_list = [([np.array([0., 1.]), np.array([0., 1.])], 0.5),
                         ([np.array([0.5, 0.73105858]), np.array([0.43316699, 0.55750901])], 0.7059216403068447)]
        message_list = ["Check you are applying logistic_layer() to the activations of the final hidden layer",
                        "Function does not use the activation_function provided as an argument"]
        
    elif function_name == "get_weights_gradients":
        delta = np.array([-1, 1])
        current_input = np.array([1, 2, 3], dtype="float64")
        prev_hidden = np.array([0.5, 0.1])
        
        args_list = [(delta, current_input, prev_hidden)]
        kwargs_list = [{}]
        expected_list = [(np.array([[-1., -1., -2., -3.], [1., 1., 2., 3.]]), np.array([[-0.5, -0.1], [0.5, 0.1]]))]
        message_list = ["Make sure you have used the calculate_weights_gradient() function correctly for both the input and recurrent weights gradients"]

    elif function_name == "get_prev_delta":
        delta = np.array([-1, 1])
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        prev_hidden = np.array([0.5, 0.1])
        
        args_list = [(delta, recurrent_weights, prev_hidden),
                     (delta, recurrent_weights, prev_hidden)]
        kwargs_list = [{},
                       {"activation_derivative": sigmoid_derivative}]
        expected_list = [np.array([-1, 0]),
                         np.array([-0.25, 0])]
        message_list = ["Incorrect",
                        "Function does not use the activation_derivative provided as an argument"]
        
    elif function_name == "get_overall_weights_gradients":
        delta = np.array([-1, 1])
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        values_list = [np.array([1, 0, -1], dtype="float64"), 
                       np.array([0, 1, 1], dtype="float64"),
                       np.array([1, 2, 3], dtype="float64")]
        prev_hidden_list = [np.array([0.1, 0.5]), 
                            np.array([0.5, 0.1])]
        
        args_list = [(delta, recurrent_weights, values_list, prev_hidden_list),
                     (delta, recurrent_weights, values_list, prev_hidden_list),
                     (delta, recurrent_weights, values_list[-1:], [])]
        kwargs_list = [{},
                       {"activation_derivative": sigmoid_derivative},
                       {}]
        expected_list = [(np.array([[-2., -1., -3., -4.], [0., 0., 2., 4.]]), np.array([[-0.6, -0.6], [0.5, 0.1]])),
                         (np.array([[-1.25, -1., -2.25, -3.25], [0.9375, 0.9375, 2., 3.0625]]), np.array([[-0.525, -0.225], [0.5, 0.1]])),
                         (np.array([[-1., -1., -2., -3.], [1., 1., 2., 3.]]), None)]
        message_list = ["Did you remember to add the last input weights gradient?",
                        "Function does not use the activation_derivative provided as an argument",
                        "Function does not handle a 1-timestep sequence"]
    
    elif function_name == "recurrent_classifier_backward":
        predicted_a = 0.5
        predicted_b = 0
        actual = 1
        output_weights = np.array([1, 1, -1])
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        values_list = [np.array([1, 2, 3], dtype="float64"),
                        np.array([0, 0, 1], dtype="float64")]
        hidden_list_a = [np.array([0, 1]),
                         np.array([0, 1])]
        hidden_list_b = [np.array([0.5, 0.1]), 
                         np.array([0.1, 0.5])]
        
        args_list = [(predicted_a, actual, output_weights, recurrent_weights, values_list, hidden_list_a),
                     (predicted_b, actual, output_weights, recurrent_weights, values_list, hidden_list_b)]
        kwargs_list = [{},
                       {"activation_derivative": sigmoid_derivative}]
        expected_list = [(np.array([[-0., -0., -0., -0.], [1., 0.5, 1., 2.]]), np.array([[-0., -0.], [0., 0.5]]), np.array([-0.5, -0., -0.5])),
                         (np.array([[-0.1525, -0.0625, -0.125, -0.2775], [0.2644, 0.0144, 0.0288, 0.2932]]), np.array([[-0.045, -0.009], [0.125, 0.025]]), np.array([-1., -0.1, -0.5]))]
        message_list = ["Incorrect",
                        "Function does not use the activation_derivative provided as an argument"]
        
    elif function_name == "train_epoch_classifier":
        train_samples = [
            ([np.array([1, 2, 3], dtype="float64"),
              np.array([0, 1, 1], dtype="float64")],
             1),
            ([np.array([1, 1, 0], dtype="float64"),
              np.array([2, 0, 1], dtype="float64")],
             0),
            ([np.array([1, 2, 0], dtype="float64"),
              np.array([-1, 1, 0], dtype="float64")],
             1)
        ]
        input_weights = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0]], dtype="float64")
        recurrent_weights = np.array([[0, 1], [-1, 1]], dtype="float64")
        output_weights = np.array([1, 1, -1], dtype="float64")
        learning_rate = 0.1
        
        args_list = [(train_samples, input_weights, recurrent_weights, output_weights, learning_rate),
                     (train_samples, input_weights, recurrent_weights, output_weights, learning_rate)]
        kwargs_list = [{},
                       {"activation_name": "sigmoid"}]
        expected_list = [(np.array([[-1.08859476, 0.82281048, 0., -0.08859476], [-1.16332899, -0.04837618, 0.75581843, -0.2]]), np.array([[0., 1.], [-1., 0.93213798]]), np.array([0.99559093, 0.91140524, -0.93897512])),
(np.array([[-9.99317140e-01, 9.69451317e-01, 1.70455648e-02, 2.97921835e-04], [-9.99910174e-01, 4.08233517e-02, 9.83614995e-01, 8.61141623e-03]]), np.array([[2.43304718e-04, 1.00343240e+00], [-9.99951070e-01, 9.96100956e-01]]), np.array([0.98157855, 0.95232173, -0.98592361]))]
        message_list = ["Incorrect",
                        "Function does not use the activation_name provided as an argument"]
        
    elif function_name == "softmax_layer":
        values = np.array([1, -1])
        weights = np.array([[1, 1, 1], [1, 0, 0], [0, 1 + np.log(2), 0]])
        
        args_list = [(values, weights)]
        kwargs_list = [{}]
        expected_list = [np.array([0.25, 0.25, 0.5])]
        message_list = ["Incorrect"]
        
    elif function_name == "get_output_sequence":
        hidden_list = [np.array([1, -1]), np.array([1, 1]), np.array([0, 0])]
        weights = np.array([[1, 1, 1], [1, 0, 0], [0, 1 + np.log(2), 0]])
        
        args_list = [(hidden_list, weights)]
        kwargs_list = [{}]
        expected_list = [[np.array([0.25, 0.25, 0.5]), np.array([0.71123459, 0.09625514, 0.19251027]), np.array([0.4223188, 0.4223188, 0.1553624])]]
        message_list = ["Incorrect"]
        
    elif function_name == "recurrent_tagger_forward":
        input_weights = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0]])
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        output_weights = np.array([[1, 1, -1], [np.log(2), 0, 0], [0, 1, 0]])
        input_values = [np.array([1, 2, 3], dtype="float64"),
                        np.array([0, 1, 1], dtype="float64")]
        
        args_list = [(input_values, input_weights, recurrent_weights, output_weights),
                     (input_values, input_weights, recurrent_weights, output_weights)]
        kwargs_list = [{},
                       {"activation_function": sigmoid}]
        expected_list = [([np.array([0., 1.]), np.array([0., 1.])], [np.array([0.25, 0.5, 0.25]), np.array([0.25, 0.5, 0.25])]),
                         ([np.array([0.5, 0.73105858]), np.array([0.43316699, 0.55750901])], [np.array([0.37158215, 0.34445923, 0.28395863]), np.array([0.40394088, 0.33655371, 0.25950541])])]
        message_list = ["Incorrect",
                        "Function does not use the activation_function provided as an argument"]
        
    elif function_name == "get_intermediate_weights_gradients":
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        output_weights = np.array([[1, 1, -1], [np.log(2), 0, 0], [0, 1, 0]])
        values_list = [np.array([1, 2, 3], dtype="float64"),
                       np.array([0, 1, 1], dtype="float64")]
        hidden_list = [np.array([0, 1], dtype="float64"), 
                       np.array([0, 1], dtype="float64")]
        outputs_list = [np.array([0.25, 0.5 , 0.25]), 
                        np.array([0.25, 0.5 , 0.25])]
        actuals_list = [np.array([0, 1, 0], dtype="float64"),
                        np.array([1, 0, 0], dtype="float64")]
        
        args_list = [(outputs_list, actuals_list, output_weights, recurrent_weights, values_list, hidden_list),
                     (outputs_list, actuals_list, output_weights, recurrent_weights, values_list, hidden_list)]
        kwargs_list = [{},
                       {"activation_derivative": tanh_derivative}]
        expected_list = [([np.array([[0., 0., 0., 0.], [-0.25, -0.25, -0.5, -0.75]]), np.array([[-0., -0., -0., -0.], [1.5, 0.75, 2.25, 3.]])], [None, np.array([[-0., -0.], [0., 0.75]])], [np.array([[0.25, 0., 0.25], [-0.5, -0., -0.5], [0.25, 0., 0.25]]), np.array([[-0.75, -0., -0.75], [0.5, 0., 0.5], [0.25, 0., 0.25]])]),
                         ([np.array([[0.5, 0.5, 1., 1.5], [-0., -0., -0., -0. ]]), np.array([[-0.5, 0., -0.5, -0.5], [0., 0., 0., 0. ]])], [None, np.array([[-0., -0.5], [0., 0. ]])], [np.array([[0.25, 0., 0.25], [-0.5, -0., -0.5 ], [0.25, 0., 0.25]]), np.array([[-0.75, -0., -0.75], [0.5, 0., 0.5 ], [0.25, 0., 0.25]])])]
        message_list = ["Incorrect",
                        "Function does not use the activation_derivative provided as an argument"]
    
    elif function_name == "recurrent_tagger_backward":
        recurrent_weights = np.array([[0, 1], [-1, 1]])
        output_weights = np.array([[1, 1, -1], [np.log(2), 0, 0], [0, 1, 0]])
        values_list = [np.array([1, 2, 3], dtype="float64"),
                       np.array([0, 1, 1], dtype="float64")]
        hidden_list = [np.array([0, 1], dtype="float64"), 
                       np.array([0, 1], dtype="float64")]
        outputs_list = [np.array([0.25, 0.5 , 0.25]), 
                        np.array([0.25, 0.5 , 0.25])]
        actuals_list = [np.array([0, 1, 0], dtype="float64"),
                        np.array([1, 0, 0], dtype="float64")]
        
        args_list = [(outputs_list, actuals_list, output_weights, recurrent_weights, values_list, hidden_list),
                     (outputs_list, actuals_list, output_weights, recurrent_weights, values_list, hidden_list)]
        kwargs_list = [{},
                       {"activation_derivative": tanh_derivative}]
        expected_list = [(np.array([[0., 0., 0., 0.], [0.625, 0.25, 0.875, 1.125]]), np.array([[0., 0.], [0., 0.75]]), np.array([[-0.25, 0., -0.25], [0., 0., 0.], [0.25, 0., 0.25]])),
                         (np.array([[0., 0.25, 0.25, 0.5 ], [0., 0., 0., 0. ]]), np.array([[0., -0.5], [0., 0. ]]), np.array([[-0.25, 0., -0.25], [0., 0., 0. ], [0.25, 0., 0.25]]))]
        message_list = ["Incorrect",
                        "Function does not use the activation_derivative provided as an argument"]
        
    elif function_name == "train_epoch_tagger":
        train_samples = [
            ([np.array([1, 2, 3], dtype="float64"),
              np.array([0, 1, 1], dtype="float64")],
             [np.array([0, 1, 0], dtype="float64"),
              np.array([1, 0, 0], dtype="float64")]),
            ([np.array([1, 1, 0], dtype="float64"),
              np.array([2, 0, 1], dtype="float64")],
             [np.array([0, 1, 0], dtype="float64"),
              np.array([0, 0, 1], dtype="float64")]),
            ([np.array([1, 2, 0], dtype="float64"),
              np.array([-1, 1, 0], dtype="float64")],
             [np.array([1, 0, 0], dtype="float64"),
              np.array([1, 0, 0], dtype="float64")])
        ]
        input_weights = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0]], dtype="float64")
        recurrent_weights = np.array([[0, 1], [-1, 1]], dtype="float64")
        output_weights = np.array([[1, 1, -1], [np.log(2), 0, 0], [0, 1, 0]], dtype="float64")
        learning_rate = 0.1
        
        args_list = [(train_samples, input_weights, recurrent_weights, output_weights, learning_rate),
                     (train_samples, input_weights, recurrent_weights, output_weights, learning_rate)]
        kwargs_list = [{},
                       {"activation_name": "sigmoid"}]
        expected_list = [(np.array([[-0.93787167, 1.07030676, 0.10789978, 0.00817844], [-1.15769891, -0.05641535, 0.75399396, -0.1125]]), np.array([[0., 1.], [-1.00156495, 0.87795962]]), np.array([[1.03779233, 0.96987765, -0.93163935], [0.67198756, -0.00875264, -0.02909756], [-0.01663271, 1.03887499, -0.03926309]])),
                         (np.array([[-1.00061805, 0.99037182, -0.00224858, -0.01330554], [-1.00769773, 0.02104869, 0.98288532, 0.006734 ]]), np.array([[0.00890734, 1.01247996], [-1.01030887, 0.98217801]]), np.array([[1.02820956, 0.99433895, -0.97256616], [0.69248191, 0.00305496, 0.00321947], [-0.02754428, 1.00260609, -0.03065331]]))]
        message_list = ["Incorrect",
                        "Function does not use the activation_name provided as an argument"]
        
    elif function_name == "tagged_sample_loss":
        values_list = [np.array([1, 2, 3]), np.array([0, 1, 1])]
        actuals_list = [np.array([0, 1, 0], dtype="float64"), np.array([1, 0, 0], dtype="float64")]
        input_weights = np.array([[-1, 1, 0, 0], [-1, 0, 1, 0]], dtype="float64")
        recurrent_weights = np.array([[0, 1], [-1, 1]], dtype="float64")
        output_weights = np.array([[1, 1, -1], [np.log(2), 0, 0], [0, 1, 0]], dtype="float64")
        
        args_list = [(values_list, actuals_list, input_weights, recurrent_weights, output_weights),
                     (values_list, actuals_list, input_weights, recurrent_weights, output_weights)]
        kwargs_list = [{},
                       {"activation_function": tanh}]
        expected_list = [1.0397207708399179,
                         1.0006215814938644]
        message_list = ["Incorrect",
                        "Function does not use the activation_function provided as an argument"]
    
    else:
        print("Unrecognized function name")
        return
    
    all_correct = True
    for (expected, message, args, kwargs) in zip(expected_list, message_list, args_list, kwargs_list):
        all_correct = check_function(function, expected, message, *args, **kwargs)
        if not all_correct:
            break
    
    if all_correct:
        print("Correct!")