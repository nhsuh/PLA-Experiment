import numpy as np
import random
import math
import matplotlib.pyplot as plt

"""
Generate an 11-dimensional weight vector w*, where the first dimension is 0 and the
other 10 dimensions are sampled independently at random from the uniform (0,1)
distribution (the first dimension will serve as the threshold and we set it to 0 for
convenience)
"""
iterations_dist = np.array([])
for _ in range(1000):
    weights_final = np.random.uniform(0, 1, 10)
    weights_final = np.insert(weights_final, 0, 0)

    """
    Generate a random training set with 100 examples, where each dimension of each train-
    ing example is sampled independently at random from the uniform {−1,1}
    distribution. 
    """
    training_set_x = np.random.uniform(-1, 1, (100, 10))
    training_set_x = np.insert(training_set_x, 0, 1, axis=1)

    """
    The examples are all classified in accordance with w*
    """
    training_set_y = np.ones(100)
    for i in range(len(training_set_x)):
        training_set_y[i] = math.copysign(1, np.dot(weights_final, training_set_x[i]))

    """
    Start perceptron learning algorithm with zero weight vector
    """
    learned_weights = np.zeros(11)

    incomplete = True #are we supposed to keep everything in there after each iteration?
    iterations = 0
    while incomplete:
        iterations += 1
        incorrect_indices = np.array([])
        incomplete = False
        for i in range(len(training_set_x)):
            guess = math.copysign(1, np.dot(learned_weights, training_set_x[i]))
            if guess != training_set_y[i]:
                incorrect_indices = np.append(incorrect_indices, [i])
                incomplete = True
        if incomplete:
            incorrect_indices = incorrect_indices.astype(int)
            rand_index = random.randint(0, len(incorrect_indices) - 1)
            rand_x = training_set_x[incorrect_indices[rand_index]]
            rand_y = training_set_y[incorrect_indices[rand_index]]
            learned_weights = learned_weights + (rand_y*rand_x)

    iterations_dist = np.append(iterations_dist, [iterations])

print(iterations_dist)
plt.hist(iterations_dist, 20, (0, 1000)) #does the histogram look right?
plt.show()

