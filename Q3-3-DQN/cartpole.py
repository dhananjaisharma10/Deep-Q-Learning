"""Parameters for DQN in CartPole environment
"""

# Environment name.
ENV = 'CartPole-v0'

# Q-Network configuration.
HIDDEN_1 = 24
HIDDEN_2 = 48
ACTIVATION = 'relu'

# Training/testing configuration.
NUM_TRAINING_EPISODES = 10000
NUM_TEST_EPISODES = 20
BATCH_SIZE = 32
EPOCHS = 1
# Test interval.
INTERVAL = 100
# Number of consecutive episodes of EARLY_STOPPING_REWARD
# after which training is stopped.
EARLY_STOPPING = 100
EARLY_STOPPING_REWARD = 200

# Hyper-parameters.
# Learning rate.
LR = 0.001
# Discounting factor.
GAMMA = 1
# Exploration config.
EPSILON = 0.5
EPSILON_MIN = 0.1
DECAY_PERIOD = 1500
# Replay memory size.
BUFFER_SIZE = 70000
# Model save dir.
WEIGHTS = './cartpole_weights'
