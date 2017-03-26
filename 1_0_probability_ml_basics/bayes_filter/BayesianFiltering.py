import time

import matplotlib.pyplot as plt
import numpy as np


def histogram_plot(pos, title=None, c='b'):
    axis = plt.gca()
    x = np.arange(len(pos))
    axis.bar(x, pos, color=c)
    plt.ylim((0, 1))
    plt.xticks(np.asarray(x) + 0.4, x)
    if title is not None:
        plt.title(title)


def normalize(prior, likelihood):
    #normalized_value = np.ones(len(prior))
    # TODO: Implement the normalization function
    return np.sum(likelihood * prior)


def compute_likelihood(map, measurement, prob_correct_measurement):
    likelihood = np.ones(len(map))
    # DONE: compute the likelihood
    for iter in range(map.size):
        if measurement == map[iter]:
            likelihood[iter] = prob_correct_measurement
        else:
            likelihood[iter] = 1-prob_correct_measurement
    return likelihood


def measurement_update(prior, likelihood):
    # TODO: compute posterior, use function normalize
    bel_post = np.ones(len(prior))
    normalize_constante = normalize(prior, likelihood)

    for xi in range(prior.size):
        bel_post[xi] = likelihood[xi]*prior[xi] / normalize_constante

    assert(np.sum(bel_post)) # Check if sum == 1
    return bel_post  # TODO: change this line to return posterior


def prior_update(posterior, movement, movement_noise_kernel):
    # TODO: compute the new prior
    # HINT: be careful with the movement direction and noise kernel!
    prior_prediciton = np.zeros(len(posterior))
    # The proability of the movment results for
    for xi in range(posterior.size):
        if movement >=0:
            prior_prediciton[xi] = posterior[(xi-movement+20)%20]*movement_noise_kernel[1] \
                                   + posterior[(xi-movement+21)%20] * movement_noise_kernel[0] \
                                   + posterior[((xi-movement+19)%20)] * movement_noise_kernel[2]
        else:
            prior_prediciton[xi] = posterior[(xi-movement+20)%20] * movement_noise_kernel[1] \
                                   + posterior[(xi-movement+19)%20] * movement_noise_kernel[0] \
                                   + posterior[(xi-movement+21)%20] * movement_noise_kernel[2]
    assert(np.sum(prior_prediciton != 1))  #Check if prob is 1
    return prior_prediciton  # TODO: change this line to return new prior


def run_bayes_filter(measurements, motions, plot_histogram=False):
    map = np.array([0] * 20) # DONE: define the map
    map[1] = map[5] = map[9] = map[10] = map[14] = map[16] = map[18] = 1 # Define doors as 1 from excercise
    sensor_prob_correct_measure = 0.9  # DONE: define the probability of correct measurement
    movement_noise_kernel = [0.15, 0.8, 0.05]  # [ undershoot, perfect, overshoot]

    # Assume uniform distribution since you do not know the starting position
    prior = np.array([1. / 20] * 20)

    number_of_iterations = len(measurements)
    if plot_histogram:
        fig = plt.figure("Bayes Filter")
    for iteration in range(number_of_iterations):
        # Compute the likelihood
        likelihood = compute_likelihood(map, measurements[iteration],
                                        sensor_prob_correct_measure)
        # Compute posterior
        posterior = measurement_update(prior, likelihood)
        if plot_histogram:
            plt.cla()
            histogram_plot(map, title="Measurement update", c='k')
            histogram_plot(posterior, title="Measurement update", c='y')
            fig.canvas.draw()
            plt.show(block=False)
            time.sleep(.5)

        # Update prior
        prior = prior_update(posterior, motions[iteration],
                             movement_noise_kernel)
        if plot_histogram:
            plt.cla()
            histogram_plot(map, title="Prior update", c='k')
            histogram_plot(prior, title="Prior update")
            fig.canvas.draw()
            plt.show(block=False)
            time.sleep(.5)
    plt.show()
    return prior
