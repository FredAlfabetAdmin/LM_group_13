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
import cv2


class Blob_Detection():
    def __init__(self, camera_width: int, camera_height: int, dark: bool = False) -> None:
        self.camera_width = camera_width
        self.camera_height = camera_height

        self.params = cv2.SimpleBlobDetector_Params()
        self.params.filterByColor = True
        self.params.blobColor = 0 if dark else 255
        self.params.filterByCircularity = True
        self.params.minCircularity = 0.6
        self.params.filterByConvexity = True
        self.params.minConvexity = 0.9
        self.params.filterByInertia = True
        self.params.minInertiaRatio = 0.6
        self.detector = cv2.SimpleBlobDetector_create(self.params)

    def blob_detect(self, rob: IRobobo):
        frame = rob.get_image_front()

        # TODO:resize if needed
        frame = cv2.resize(frame, (self.camera_width, self.camera_height))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        # mask for the green color
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        # AND operation to get the green regions
        green_regions = cv2.bitwise_and(frame, frame, mask=green_mask)

        gray_frame = cv2.cvtColor(green_regions, cv2.COLOR_BGR2GRAY)

        # perform blob detection
        keypoints = self.detector.detect(gray_frame)

        cv2.imwrite(str("./frame.png"), frame)
        cv2.imwrite(str("./gray_frame.png"), gray_frame)

        x, y = 0, 0.5
        size_percent = gray_frame[gray_frame > 0.5].shape[0] / (gray_frame.shape[0] * gray_frame.shape[1]) * 100

        if keypoints:
            keypoint = keypoints[0]
            x, y = int(keypoint.pt[0]) / self.camera_width, int(keypoint.pt[1]) / self.camera_height
            # size_percent = (keypoint.size / (self.camera_width * self.camera_height)) * 100
            # x and y values along with the percentage of blob area
        return [x, y, size_percent]


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


def get_current_state(scaler, rob):
    green_perc = Blob_Detection(640, 480).blob_detect(rob)[-1]
    front_sensors, back_sensors = scale_and_return_ordered(scaler, rob.read_irs())
    #print(front_sensors)
    state = tuple(1 if value > -0.3 else 0 for value in front_sensors)
    return state + (green_perc,)


def scale_and_return_ordered(scaler, irs):
    irs = scaler.transform([irs])[0].tolist()
    # Front sensors fromt left to right!! Not default order
    front_sensors = [irs[7], irs[2], irs[4], irs[3], irs[5]]
    # Back sensors from left to right
    back_sensors = [irs[0], irs[6], irs[1]]
    return front_sensors,back_sensors
    # front_sensors = [FrontLL, FrontL, FrontC, FrontR, FrontRR]
    # back_sensors = [BackL, BackC, BackR]


def eucl_distance(point1, point2):
    delta1 = (point1[0] - point2[0])**2
    delta2 = (point1[1] - point2[1])**2
    total = delta1 + delta2
    distance = np.sqrt(total)
    return distance

def move_robobo_and_calc_reward(scaler, action, rob, state):
    extra_reward = 0

    if isinstance(rob, SimulationRobobo):
        if action == 0:  # Move forward
            print("So going forward")
            movement = [50, 50, 250]
        elif action == 1:  # Move left
            print("So going left")
            movement = [-50, 50, 250]
        elif action == 2:  # Move right
            print("So going right")
            movement = [50, -50, 250]
        pos_before = rob.position()
        rob.move_blocking(int(movement[0]), int(movement[1]), int(movement[2]))
        pos_after = rob.position()
        distance = eucl_distance([pos_before.x, pos_before.y], [pos_after.x, pos_after.y])

        # If robot actually moves forward give it a reward
        if distance > 0.04:
            extra_reward += 2
    else:
        if action == 0:  # Move forward
            movement = [20, 20, 250]
        elif action == 1:  # Move left
            print("Turning left")
            movement = [-20, 20, 250]
        elif action == 2:  # Move right
            print("Turning right")
            movement = [20, -20, 250]
        rob.move_blocking(int(movement[0]), int(movement[1]), int(movement[2]))
    next_state = get_current_state(scaler, rob)

    # If robobo turned away from object
    if sum(state) > sum(next_state) and action != 0:
        extra_reward += 3 + sum(state) - sum(next_state)

    reward = extra_reward
    return reward, next_state


def run_qlearning_classification(rob: IRobobo):
    print('connected')

    num_actions = 3  # Number of possible actions

    # Hardware test run
    if not isinstance(rob, SimulationRobobo):
        agent = QLearningAgent(num_actions, exploration_prob=0)
        scaler = joblib.load('hardware_powertrans_scaler.gz')
        state = get_current_state(scaler, rob)
        while True:  # Take max 75 steps per round
            action = agent.choose_action(state)

            # Simulate taking the chosen action and observe the next state and reward
            print(state)
            reward, next_state = move_robobo_and_calc_reward(scaler, action, rob, state)  # Replace with your game logic
            print("Reward:", reward)

            # Move to the next state for the next iteration
            state = next_state
        return

    # Simulation training
    scaler = joblib.load('software_powertrans_scaler.gz')
    agent = QLearningAgent(num_actions, exploration_prob=0)
    rob.play_simulation()
    for round in range(150):
        print(f"-=-=-=-=-=-=- Round {round} -=-=-=-=-=-=-=-=-=-")
        if round > 3: #Only create random positions after 3 rounds
            rob.set_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.03), Orientation(-90, -90, -90))
            rob.set_target_position(Position((random.random()*2.4)-1.2, (random.random()*2.4)-1.2, 0.2))
        rob.play_simulation()
        state = get_current_state(scaler,rob)  # Replace with your function to get the current state
        for step in range(240):  # Take max 75 steps per round
            action = agent.choose_action(state)

            # Simulate taking the chosen action and observe the next state and reward
            print(state)
            reward, next_state = move_robobo_and_calc_reward(scaler, action, rob, state)  # Replace with your game logic

            # Update Q-value based on the observed reward and the Q-learning update rule
            print("Reward:", reward)
            agent.update_q_value(state, action, reward, next_state)

            # Move to the next state for the next iteration
            state = next_state
        print(agent.q_table)
