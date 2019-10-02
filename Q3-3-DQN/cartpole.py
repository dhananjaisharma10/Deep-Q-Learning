"""Parameters for DQN in CartPole environment
"""

ENV = 'CartPole-v0'

HIDDEN_1 = 24
HIDDEN_2 = 48

NUM_TRAINING_EPISODES = 10000
NUM_TEST_EPISODES = 100

LR = 0.001
GAMMA = 1
EPSILON = 1
EPSILON_MIN = 0.05
DECAY_PERIOD = 1000
BATCH_SIZE = 32
EPOCHS = 20

BUFFER_SIZE = 20000

WEIGHTS = './cartpole_weights'
INTERVAL = 10
