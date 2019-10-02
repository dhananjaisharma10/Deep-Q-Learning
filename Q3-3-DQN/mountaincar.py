"""Parameters for DQN in MountainCar environment
"""

ENV = 'MountainCar-v0'

NUM_TRAINING_EPISODES = 10000
NUM_TEST_EPISODES = 100

HIDDEN_1 = 20
HIDDEN_2 = 10
LR = 0.0001
GAMMA = 1
EPSILON = 0.5
EPSILON_DECAY = (0.5 - 0.05) / 100000
BATCH_SIZE = 32
EPOCHS = 10
WEIGHTS = './mountaincar_weights'
INTERVAL = 1
