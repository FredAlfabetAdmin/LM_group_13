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
    def __init__(self, num_actions, learning_rate=0.1, discount_factor=0.9, exploration_prob=0.0):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_prob = exploration_prob

        # Q-table to store Q-values for each state-action pair
        self.q_table = {((0, 0, 0, 0, 0), 0): 51.34152947264171, ((0, 0, 0, 0, 0), 2): 47.885198887833546, ((0, 0, 0, 0, 0), 1): 51.35592956589, ((0, 0, 0, 0, 1), 0): 31.791309250369554, ((0, 0, 1, 0, 1), 0): 32.38579216688231, ((0, 0, 1, 1, 1), 0): 32.55263074816037, ((1, 1, 1, 1, 1), 0): 33.40038355201896, ((1, 1, 1, 1, 1), 1): 24.123466656007597, ((0, 1, 1, 1, 1), 0): 16.415279564318787, ((1, 1, 1, 1, 1), 2): 40.864594250096815, ((1, 1, 1, 0, 0), 0): 36.55612680889666, ((0, 0, 1, 1, 1), 2): 23.15886917822469, ((1, 1, 1, 0, 0), 2): 47.801525485225525, ((0, 0, 1, 1, 1), 1): 20.557963944089483, ((1, 1, 1, 1, 0), 0): 36.2728283733809, ((1, 1, 1, 0, 0), 1): 31.799834605499093, ((1, 1, 1, 1, 0), 2): 31.57915195844804, ((0, 1, 1, 1, 1), 1): 4.7248174103738885, ((1, 1, 0, 1, 1), 0): 0.9224595946589913, ((1, 0, 1, 0, 0), 0): 40.75424790731623, ((1, 0, 0, 0, 0), 0): 51.443249629558295, ((1, 0, 0, 0, 0), 1): 43.791572396757616, ((1, 1, 1, 1, 0), 1): 15.620413244181048, ((0, 0, 0, 1, 1), 0): 24.97835496555777, ((1, 1, 0, 0, 0), 0): 42.52889689835159, ((0, 0, 1, 0, 0), 0): 35.2641949619424, ((1, 0, 1, 0, 0), 2): 27.306357159370616, ((1, 0, 0, 0, 0), 2): 51.940650603051324, ((1, 1, 1, 0, 1), 1): 25.053302420508444, ((1, 1, 0, 0, 0), 1): 25.53219228823734, ((0, 1, 1, 1, 0), 0): 13.192229690622792, ((0, 0, 1, 1, 0), 0): 23.530546945311308, ((0, 0, 1, 0, 0), 1): 19.275457285110583, ((0, 0, 0, 1, 0), 0): 0.5334365040925576, ((0, 1, 1, 0, 0), 0): 0.4, ((0, 1, 1, 0, 0), 1): 17.588119936031017, ((1, 1, 0, 0, 0), 2): 44.47216665337521, ((0, 0, 0, 1, 1), 2): 10.912454845487739, ((0, 0, 0, 1, 1), 1): 6.919188516892165, ((0, 0, 1, 1, 0), 1): 3.0931998777737397, ((0, 0, 0, 0, 1), 1): 49.640214445769296, ((0, 1, 1, 1, 0), 1): 1.5483052224728775, ((0, 0, 0, 1, 0), 1): 6.985808416497067, ((0, 0, 1, 0, 1), 2): 14.505215992874067, ((0, 1, 0, 0, 0), 0): 3.02912284204857, ((0, 1, 1, 1, 0), 2): 3.0319734486310335, ((0, 0, 1, 0, 1), 1): 3.329497994276941, ((0, 0, 1, 0, 0), 2): 28.094945959664642, ((0, 1, 1, 1, 1), 2): 31.4482077907151, ((1, 0, 1, 1, 1), 0): 19.146039626991538, ((0, 0, 0, 0, 1), 2): 13.458025181566569, ((1, 0, 1, 0, 1), 1): 10.159304189425045, ((1, 1, 1, 0, 1), 2): 4.90296366487878, ((1, 0, 1, 0, 0), 1): 5.2150992789125485, ((1, 0, 1, 0, 1), 0): 2.1860050390596304, ((0, 0, 1, 1, 0), 2): 4.372686036368711, ((1, 0, 0, 0, 1), 0): 1.3848521609529583, ((1, 1, 1, 0, 1), 0): 3.210385901795638, ((1, 0, 1, 1, 1), 1): 2.769125637706712, ((1, 1, 0, 0, 1), 0): 4.2792568017580335, ((1, 0, 0, 1, 1), 2): 3.588113991472497}

    def get_q_value(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        # Exploration-exploitation trade-off
        if np.random.rand() < self.exploration_prob:
            return np.random.choice(self.num_actions)  # Explore
        else:
            print("-- delib --")
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


def get_current_state(scaler, irs):
    front_sensors, back_sensors = scale_and_return_ordered(scaler, irs)
    state = tuple(1 if value > 0.2 else 0 for value in front_sensors)
    return state


def scale_and_return_ordered(scaler, irs):
    irs = scaler.transform([irs])[0].tolist()
    # Front sensors fromt left to right!! Not default order
    front_sensors = [irs[7], irs[2], irs[4], irs[3], irs[5]]
    # Back sensors from left to right
    back_sensors = [irs[0], irs[6], irs[1]]
    return front_sensors,back_sensors
    # front_sensors = [FrontLL, FrontL, FrontC, FrontR, FrontRR]
    # back_sensors = [BackL, BackC, BackR]


def move_robobo_and_calc_reward(scaler, action, rob):
    forward_reward = 0
    if action == 0: #Move forward
        print("So going forward")
        forward_reward = 1
        movement = [50, 50, 250]
    elif action == 1: #Move left
        print("So going left")
        movement = [-50, 50, 250]
    elif action == 2: #Move right
        print("So going right")
        movement = [50, -50, 250]
    rob.move_blocking(int(movement[0]), int(movement[1]), int(movement[2]))
    state = get_current_state(scaler, rob.read_irs())
    reward = 5 - sum(state) + forward_reward
    return reward, state


def run_qlearning_classification(rob: IRobobo):
    print('connected')

    num_actions = 3  # Number of possible actions
    agent = QLearningAgent(num_actions)

    # Hardware test run
    if not isinstance(rob, SimulationRobobo):
        scaler = joblib.load('hardware_powertrans_scaler.gz')
        state = get_current_state(scaler, rob.read_irs())
        for step in range(240):  # Take max 75 steps per round
            action = agent.choose_action(state)

            # Simulate taking the chosen action and observe the next state and reward
            print(state)
            reward, next_state = move_robobo_and_calc_reward(scaler, action,rob)  # Replace with your game logic
            print("Reward:", reward)

            # Move to the next state for the next iteration
            state = next_state
        return

    # Simulation training
    scaler = joblib.load('software_powertrans_scaler.gz')
    rob.play_simulation()
    for round in range(150):
        print(f"-=-=-=-=-=-=- Round {round} -=-=-=-=-=-=-=-=-=-")
        if round > 3: #Only create random positions after 3 rounds
            rob.set_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.03), Orientation(-90, -90, -90))
            rob.set_target_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.2))
        rob.play_simulation()
        state = get_current_state(scaler, rob.read_irs())  # Replace with your function to get the current state
        for step in range(240):  # Take max 75 steps per round
            action = agent.choose_action(state)

            # Simulate taking the chosen action and observe the next state and reward
            print(state)
            reward, next_state = move_robobo_and_calc_reward(scaler, action,rob)  # Replace with your game logic

            # Update Q-value based on the observed reward and the Q-learning update rule
            print("Reward:", reward)
            agent.update_q_value(state, action, reward, next_state)

            # Move to the next state for the next iteration
            state = next_state
        print(agent.q_table)
