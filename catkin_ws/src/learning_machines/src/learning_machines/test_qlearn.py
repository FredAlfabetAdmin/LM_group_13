from data_files import FIGRURES_DIR
from robobo_interface import (
    IRobobo,
    Emotion,
    LedId,
    LedColor,
    SoundEmotion,
    SimulationRobobo,
)
from robobo_interface.datatypes import (
    Acceleration,
    Position,
    Orientation,
)

from .lstm import eucl_loss_fn, LSTM
from torch import optim, nn
import torch, time, random
import numpy as np
import joblib


class QLearningAgent:
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.2):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Q-table to store Q-values for each state-action pair
        self.q_table = {}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.num_actions)  # Explore
        else:
            # Exploit - choose action with the highest Q-value
            q_values = [self.get_q_value(state, a) for a in range(self.num_actions)]
            return np.argmax(q_values)

    def update_q_value(self, state, action, reward, next_state):
        # Q-learning update rule
        current_q = self.get_q_value(state, action)
        max_future_q = max([self.get_q_value(next_state, a) for a in range(self.num_actions)])
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_future_q - current_q)

        # Update Q-value in the Q-table
        #old = self.get_q_value(state, action)
        #print(f"{state}:{action} = {old} to {new_q}")
        self.q_table[(state, action)] = new_q


scaler = joblib.load('software_powertrans_scaler.gz')


def get_current_state(irs):
    irs = scaler.transform([irs])[0].tolist()
    left_sensors = [irs[7], irs[2], irs[4]]
    right_sensors = [irs[4], irs[3], irs[5]]
    back_sensors = [irs[0],irs[6],irs[1]]
    if all(sensor > 0.15 for sensor in left_sensors + right_sensors):
        return 0
    if any(sensor > 0.15 for sensor in left_sensors):
        return 1  # State 1
    elif any(sensor > 0.15 for sensor in right_sensors):
        return 2  # State 2
    elif any(sensor > 0.15 for sensor in back_sensors):
        return 3
    else:
        return 4


def move_robobo_and_calc_reward(action, rob):
    forward_reward = 0
    if action == 0: #Move forward
        forward_reward = 0.5
        movement = [50, 50, 250]
    elif action == 1: #Move backward
        movement = [-50, -50, 250]
    elif action == 2: #Move left
        movement = [-50, 50, 250]
    elif action == 3: #Move right
        movement = [50, -50, 250]
    rob.move_blocking(int(movement[0]), int(movement[1]), int(movement[2]))
    irs = scaler.transform([rob.read_irs()])[0].tolist()
    reward = 3 - sum(irs) * 0.8 + forward_reward
    return reward, get_current_state(rob.read_irs())


def run_qlearning_classification(rob: IRobobo):
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    print('connected')
    # Example usage:
    num_actions = 4  # Number of possible actions
    agent = QLearningAgent(num_actions)

    # Simulate a game loop
    for round in range(20):
        if round > 3: #Only create random positions after 3 rounds
            rob.set_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.03), Orientation(-90, -90, -90))
            rob.set_target_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.2))
        rob.play_simulation()
        state = get_current_state(rob.read_irs())  # Replace with your function to get the current state
        for step in range(240):  # Take max 75 steps per round
            action = agent.choose_action(state)

            # Simulate taking the chosen action and observe the next state and reward
            reward, next_state = move_robobo_and_calc_reward(action,rob)  # Replace with your game logic

            # Update Q-value based on the observed reward and the Q-learning update rule
            print("Reward:", reward)
            agent.update_q_value(state, action, reward, next_state)

            # Move to the next state for the next iteration
            state = next_state
        print(agent.q_table)
