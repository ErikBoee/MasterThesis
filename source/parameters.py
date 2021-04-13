import numpy as np

#Create problem
NO_ANGLES = 16
ANGLES = np.linspace(0, np.pi, NO_ANGLES)[:-1]
LAMDA = 100
MAX_LAMDA = 1000000
C_1 = 0.001
C_2 = 0.9
TAU = 0.1
BETA = 0.5
NOISE_SIZE = 0.0

#Run
NUMBER_OF_FULL_LOOPS = 6
MAX_ITER_BFGS = 200
IMAGE_FREQUENCY = 20