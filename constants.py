
import numpy as np

NUM_STEPS = 10000

GRAVITY_X = 0
GRAVITY_Y = 0
GRAVITY_Z = -9.8

amplitudeFront = np.pi / 3
frequencyFront = 20
phaseOffsetFront = 0

amplitudeBack = np.pi / 4
frequencyBack = 20
phaseOffsetBack = np.pi

MAX_FORCE = 50

SLEEP_TIME = 1/60

numberOfGenerations = 350
populationSize = 10

#numberOfGenerations = 1
#populationSize = 1

numSensorNeurons = 9
numMotorNeurons = 8

motorJointRange = .4