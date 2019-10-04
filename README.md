# HW2 RL
Homework 2 for Deep Reinforcement Learning and Control

This repository implements [DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf), and Value Iteration and Policy Iteration as given in [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/bookdraft2017nov5.pdf).

# DQN algorithm
The subfolder `Q3-3-DQN/` consists of the modules: `DQN_Implementation.py`, `cartpole.py` and `mountaincar.py`.

`DQN_Implementation.py` implements the Deep Q-Network algorithm. It consists of three classes:

- QNetwork: implements the Q learning algorithm by using a neural network for
approximating the Q value function.
- Replay_Memory: implements the replay buffer used to train the DQN agent.
- DQN_Agent: implements the DQN algorith by combining Q learning with replay buffer

The module takes as argument the environment name as given in OpenAI gym. For example: `CartPole-v0` and `MountainCar-v0`.

`cartpole.py` and `mountaincar.py` are the configuration files for environments `CartPole-v0` and `MountainCar-v0`, respectively.

In order to run the code, please type the following command in terminal:
```
python Q3-3-DQN/DQN_Implementation.py
```

The module will also generate videos of the environment at scheduled times as specified in the configuration.

# Value Iteration and Policy Iteration

The subfolder `Q2-VI-PI\deeprl_hw2q2` consists of modules `lake_envs.py` and `rl.py`.

`rl.py` consists of various functions that implement policy iteration by combining policy evaluation and policy improvement. It also implements the value iteration algorithm. Both policy iteration and value iteration have been implemented using synchronous and asynchronous methods. Amongst asynchronous methods, there are two methods: 
- ordered states: states are iterated from smallest to highest.
- random permutation: states are iterated over in a random fashion.
- custom: only for value iteration. We implement value iteration using Manhattan Distance ordering of the states.

To run the code, please specify the configuration to be tested in the function `run_my_policy` in the module `Q2-VI-PI/runner.py`. Then run it by typing the following code in terminal:
```
python Q2-VI-PI/runner.py
```
