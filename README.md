# HW2 RL
Homework 2 for Deep Reinforcement Learning and Control

This repository implements Value Iteration, Policy Iteration

# DQN algorithm
The subfolder Q3-3-DQN consists of modules `DQN_Implementation.py` which implements the Deep Q-Network algorithm. It consists of three classes:

- QNetwork: implements the Q learning algorithm by using a neural network for
approximating the Q value function.
- Replay_Memory: implements the replay buffer used to train the DQN agent.
- DQN_Agent: implements the DQN algorith by combining Q learning with replay buffer

The module takes as argument the environment name as given in OpenAI gym. For example: CartPole-v0 and MountainCar-v0
There are two other modules named `cartpole.py` and `mountaincar.py` which act as
configuration files for environments CartPole-v0 and MountainCar-v0, respectively.

In order to run the code, 
