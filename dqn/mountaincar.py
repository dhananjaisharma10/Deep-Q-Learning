"""Parameters for DQN in MountainCar environment
"""

# Environment name.
ENV = 'MountainCar-v0'

# Q-Network configuration.
HIDDEN_1 = 32
HIDDEN_2 = 64
ACTIVATION = 'relu'

# Training/testing configuration.
NUM_TRAINING_EPISODES = 10000
NUM_TEST_EPISODES = 20
BATCH_SIZE = 32
EPOCHS = 5
# Test interval
INTERVAL = 100
# Number of episodes of getting >= EARLY_STOPPING_REWARD
# after which training is stopped.
EARLY_STOPPING = 100
EARLY_STOPPING_REWARD = -120

# Hyper-parameters
# Learning rate
LR = 0.0001
# Discounting factor
GAMMA = 1
# Exploration config
EPSILON = 1
EPSILON_MIN = 0.1
DECAY_PERIOD = 3000

# Replay memory size
BUFFER_SIZE = 70000

# Model save dir
WEIGHTS = './mountaincar_weights'

