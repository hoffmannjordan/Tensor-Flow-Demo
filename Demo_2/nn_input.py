import numpy as np

def nn_input(key):
    solution_data = training_data(key)

    window_size = solution_data.shape

    X_main = np.zeros((window_size[0], window_size[1], FEATURES))


    X_side = np.zeros(SIDE_FEATURES)
    X_side[0] = 

    Y = 

    return [X_main, X_side], Y

