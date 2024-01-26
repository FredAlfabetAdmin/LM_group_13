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
import torch.nn.functional as F
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

        #TODO:resize if needed
        frame = cv2.resize(frame, (self.camera_width, self.camera_height))
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40])
        upper_green = np.array([80, 255, 255])

        #mask for the green color
        green_mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        #AND operation to get the green regions
        green_regions = cv2.bitwise_and(frame, frame, mask=green_mask)

        gray_frame = cv2.cvtColor(green_regions, cv2.COLOR_BGR2GRAY)

        #perform blob detection
        keypoints = self.detector.detect(gray_frame)

        cv2.imwrite(str("./frame.png"), frame)
        cv2.imwrite(str("./gray_frame.png"), gray_frame)
        
        x, y = 0, 0.5
        size_percent = gray_frame[gray_frame > 0.5].shape[0] / (gray_frame.shape[0] * gray_frame.shape[1]) * 100

        if keypoints:
            keypoint = keypoints[0]
            x, y = int(keypoint.pt[0]) / self.camera_width, int(keypoint.pt[1]) / self.camera_height
            # size_percent = (keypoint.size / (self.camera_width * self.camera_height)) * 100
            #x and y values along with the percentage of blob area
        return [x, y, size_percent]  

class FoodDetect():
    def __init__(self):
        '''class to see if food is found and decay this over time'''
        max_len = 20
        self.highest_num = 0
        self.food = [0] * max_len
        self.mask = list(reversed([1-(x*(1/(max_len))) for x in range(max_len)]))

    def add_food(self, food: int) -> int:
        # Add nr_food to list and apply the mask
        found_food = food - self.highest_num # Get difference between the two and see if food has changed
        self.food = self.food[1:] #Tick the buffer one over
        self.food += [found_food] #Pytonic combine them for the class
        if food > self.highest_num: #Check if found food is higher so 0 can enter the list
            self.highest_num = food
        return (-np.sum([x*y for x,y in zip(self.food, self.mask)]) + 1 ) * (10 * (1-(self.highest_num/7))) #Apply the mask and sum.

class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Using tanh for output to handle actions in range [-1, 1]
        return x
    
class RobotEnvironment:
    def __init__(self, rob: IRobobo):
        self.rob = rob
        self.scaler = joblib.load('software_powertrans_scaler.gz')
        self.detector = Blob_Detection(640, 480)
        self.repr_trackr = 0

    def get_state(self):
        self.detected = self.detector.blob_detect(self.rob)[-1]
        irs = self.scaler.transform([self.rob.read_irs()])[0].tolist()
        front_sensors = [irs[7], irs[2], irs[4], irs[3], irs[5]]
        state = [1 if value > -0.3 else 0 for value in front_sensors]
        return np.array(state + [self.detected])

    def take_action(self, action):
        if np.random.random() < 0.7:
            choose_action = np.random.choice([0,1,2], p=action)
        else:
            choose_action = np.argmax(action)
        if choose_action == 0: #Move forward
            action = [50, 50, 250]
        elif choose_action == 1: #Move left
            action = [-50, 50, 125]
        elif choose_action == 2: #Move right
            action = [50, -50, 125]
        self.rob.move_blocking(int(action[0]), int(action[1]), int(action[2]))
        return choose_action

    def get_reward(self):
        diff = eucl_loss_fn([self.rob.position().x, self.rob.position().y], [self.rob.position().x, self.rob.position().y])
        if diff < 0.05:
            self.repr_trackr += 1
            penality = (self.repr_trackr ** 1.1)
        else:
            penality = 0
            self.repr_trackr = 0
        return [((50 -self.detector.blob_detect(self.rob)[-1]) /5)+ self.detect.add_food(self.rob.nr_food_collected()) + penality]
    
    def start_env(self):
        self.rob.play_simulation()
        self.rob.set_phone_tilt_blocking(105, 100)
        self.detect = FoodDetect()
        self.repr_trackr = 0

    def stop_env(self):
        self.rob.stop_simulation()

class RLAgent:
    def __init__(self, state_size, action_size):
        self.policy_network = PolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=0.05)
        self.gamma = 0.99  # Discount factor
        self.epsilon_clip = 0.2  # PPO clipping parameter

    def get_action(self, state):
        state = torch.from_numpy(state).float()
        action_probs = self.policy_network(state)
        action = nn.functional.softmax(action_probs, dim=0)
        return action.detach().numpy()

    def update_policy(self, states, actions, rewards):
        self.optimizer.zero_grad()
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards[::-1])

        # Compute discounted rewards
        discounted_rewards = []
        running_add = 0
        for r in rewards:
            running_add = running_add * self.gamma + r
            discounted_rewards.insert(0, running_add)

        # Normalize discounted rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / ((discounted_rewards.std() + 1e-8) if discounted_rewards.shape[0] > 1 else 1e-8)
        # Convert actions to probabilities using the policy network
        action_probs = self.policy_network(states)
        action_distribution = torch.distributions.Normal(action_probs, torch.tensor(0.1))
        log_probs = action_distribution.log_prob(actions).sum(dim=0)

        # Compute advantages
        advantages = discounted_rewards - log_probs.detach()

        # PPO Surrogate Objective
        new_action_probs = self.policy_network(states)
        new_action_distribution = torch.distributions.Normal(new_action_probs, torch.tensor(0.1))
        new_log_probs = new_action_distribution.log_prob(actions).sum(dim=0)

        ratio = torch.exp(new_log_probs - log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon_clip, 1 + self.epsilon_clip) * advantages

        loss = -torch.min(surr1, surr2).mean()

        # Update the policy using the optimizer
        loss.backward()
        self.optimizer.step()
        return loss.item()

def train_agent(agent: RLAgent, env: RobotEnvironment, num_episodes=1, max_steps_per_episode=100):
    foods, acts, rws, losses = [], [], [], []
    for episode in range(num_episodes):
        env.start_env()
        state = env.get_state()
        total_reward = []
        food, act, rw, lossl = [], [], [], []
        for _ in range(max_steps_per_episode):
            action = agent.get_action(state)
            acti = env.take_action(action)
            reward = env.get_reward()
            total_reward += reward
            print("-=-=-=-=-")
            print(state)
            print(action)
            print(reward)

            # Store the experience for updating the policy
            loss = agent.update_policy(state, action, reward)

            state = env.get_state()

            food.append(str(env.rob.nr_food_collected()))
            act.append(str(acti))
            rw.append(str(reward[0]))
            lossl.append(str(loss))
            print(f"Episode {episode + 1}, Step: {_}, Total Reward: {reward[0]}, action: {acti}")
        foods.append(' '.join(food))
        acts.append(' '.join(act))
        rws.append(' '.join(rw))
        losses.append(' '.join(lossl))
        print(f"Episode {episode + 1}, Total Reward: {np.mean(total_reward)}")
        with open(f'./res_foods_{episode}.txt', "w") as file_:
            file_.writelines(foods[-1])
        with open(f'./res_actions_{episode}.txt', "w") as file_:
            file_.writelines(acts[-1])
        with open(f'./res_rewards_{episode}.txt', "w") as file_:
            file_.writelines(rws[-1])
        with open(f'./res_losses_{episode}.txt', "w") as file_:
            file_.writelines(losses[-1])
        env.stop_env()
        torch.save(agent.policy_network.state_dict(), f'./model_{episode}.ckpt')
        time.sleep(0.25)
    with open('./res_foods.txt', "w") as file_:
        file_.writelines(foods)
    with open('./res_actions.txt', "w") as file_:
        file_.writelines(acts)
    with open('./res_rewards.txt', "w") as file_:
        file_.writelines(rws)
    with open('./res_losses.txt', "w") as file_:
        file_.writelines(losses)

def calc_loss(food_and_time: torch.Tensor, max_time: int, time_penalty: int, food_detect: FoodDetect):
    if food_and_time[0] > 0:
        n_food_reward = -torch.ceil(food_and_time[0]*(torch.log10(food_and_time[0]))) + 6
    else:
        n_food_reward = food_and_time[0] + 7
    sim_time = ((torch.pow(food_and_time[1], 2)*(1/max_time))) / max_time * time_penalty
    c_food = food_detect.add_food(food_and_time[0])
    return c_food + sim_time # n_food_reward

def run_lstm_classification(
        rob: IRobobo, 
        max_time=1.5*60*1000, time_penalty=100, 
        seq_len=128, features=6, hidden_size=128, num_outputs=3, num_layers=1,
        eval_=False):
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    # with torch.autograd.detect_anomaly():
    print('connected')

    agent = RLAgent(features, num_outputs)
    environment = RobotEnvironment(rob)

    # Eval model in hw
    if not isinstance(rob, SimulationRobobo) or eval_:
        agent.policy_network.load_state_dict(torch.load('./model_134.ckpt'))
        agent.policy_network.eval()
        with torch.no_grad():
            rob.set_phone_tilt_blocking(105, 100) #Angle phone forward
            while True:
                action = agent.get_action(state)
                acti = env.take_action(action)
        return

    max_steps_per_episode = 100  # Adjust as needed
    num_episodes = 1000  # Adjust as needed

    train_agent(agent, environment, num_episodes, max_steps_per_episode)

    torch.save(agent.policy_network.state_dict(), './model.ckpt')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()
