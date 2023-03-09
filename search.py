
import os
import numpy as np
import constants as c
import sys

from hillclimber import HILL_CLIMBER
from parallelHillClimber import PARALLEL_HILL_CLIMBER

import matplotlib.pyplot as plt

if __name__ == '__main__':

    trials = []
    for size, col in zip([.5, 1, 2], ['red', 'blue', 'green']):
        best_fitnesses = []
        for i in range(1, 6):
            np.random.seed(i)
            phc = PARALLEL_HILL_CLIMBER(size)
            phc.Evolve()
            phc.Show_Best()
            best_fitnesses.append(phc.best_fitnesses[-1])

            if i == 1:
                plt.plot([j + 1 for j in range(c.numberOfGenerations)], phc.best_fitnesses, color=col, label="Max Size: {}".format(size))
            else:
                plt.plot([j + 1 for j in range(c.numberOfGenerations)], phc.best_fitnesses, color=col)

        trials.append(best_fitnesses)


    plt.title("Peak Fitness at Each Generation")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid()

    plt.savefig("best_fitnesses.jpg")

    for trial, size in zip(trials, [.5,1, 2]):
        print("STATS for Max Size: {}".format(size))
        print(trial)
        print("Mean: {}".format(np.mean(trial)))
        print("Max: {}".format(np.max(trial)))
        print("Std: {}".format(np.std(trial)))
        print("-----------------------")