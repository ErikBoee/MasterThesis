import numpy as np

#Create problem
#NO_ANGLES = 16
#ANGLES = np.linspace(0, np.pi, NO_ANGLES + 1)[:-1]
ANGLES = np.linspace(np.pi/16*6, np.pi*10/16, 5)
NO_ANGLES = 5
LAMDA = 100
MAX_LAMDA = 1000000
C_1 = 0.001
C_2 = 0.9
TAU = 0.1
BETA = 0.0*NO_ANGLES
NOISE_SIZE = 0.15

#Run
NUMBER_OF_FULL_LOOPS = 1
MAX_ITER_BFGS = 5
IMAGE_FREQUENCY = 1