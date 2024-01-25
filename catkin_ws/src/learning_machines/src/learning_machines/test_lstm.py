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

        if keypoints:
            keypoint = keypoints[0]
            x, y = int(keypoint.pt[0]), int(keypoint.pt[1])
            size_percent = (keypoint.size / (self.camera_width * self.camera_height)) * 100
            #x and y values along with the percentage of blob area
            return x, y, size_percent
        else:
            pass
        time.sleep(3)

def move_robobo(movement, rob):
    movement = movement.detach().numpy() #Use clone since detach doesnt properly work
    movement = np.argmax(movement)
    if movement == 0: #Move forward
        movement = [50, 50, 250]
    elif movement == 1: #Move backward
        movement = [-50, -50, 250]
    elif movement == 2: #Move left
        movement = [-50, 50, 250]
    elif movement == 3: #Move right
        movement = [50, -50, 250]
    rob.move_blocking(int(movement[0]), int(movement[1]), int(movement[2]))

class FoodDetect():
    def __init__(self):
        '''class to see if food is found and decay this over time'''
        max_len = 10
        self.highest_num = 0
        self.food = torch.zeros((max_len,), dtype=torch.float32, requires_grad=True)
        self.mask = torch.tensor([1-(x*(1/(max_len))) for x in range(max_len)], dtype=torch.float32, requires_grad=True)
        tot_ = torch.sum(self.mask.detach())

    def add_food(self, food: int) -> int:
        # Add nr_food to list and apply the mask
        food -= self.highest_num #If found one it gets increased by 1 or 2 otherwise it will be 0
        self.food = self.food[1:] #Tick the buffer one over
        self.food = torch.cat([self.food, food.unsqueeze(0)])
        if self.highest_num > food:
            self.highest_num = food
        print('food_calc', self.food * self.mask)
        return (-torch.sum(self.food * self.mask) + 1 ) * 10 #Apply the mask and sum.

class RobFN(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, rob, start):
        ctx.save_for_backward(input)
        move_robobo(input, rob)
        food_and_time = torch.tensor([rob.nr_food_collected(), start - time.time()], dtype=torch.float32, requires_grad=True)
        return food_and_time

    @staticmethod
    def backward(ctx, grad_output):
        # ctx.saved_tensors contains the input from the forward pass
        input, = ctx.saved_tensors
        grad_input = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.full((4,), torch.mean(grad_output)) #Use 1 for the rob_pos grad, *100 since the output of the model is like that
        return grad_input, None, None

def evaluation(rob, model: nn.Module, scaler: joblib.load, seq: torch.Tensor):
    model.load_state_dict(torch.load('./model.ckpt'))
    model.eval()
    with torch.no_grad():
        for _ in range(200): #Take max 75 steps per round
            robfn = RobFN.apply #Define the custom grad layer
            # Get the input data
            orientation = rob.read_orientation()
            accelleration = rob.read_accel()
            x = torch.tensor(scaler.transform([rob.read_irs()])[0].tolist() + [orientation.yaw, orientation.pitch, orientation.roll] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
            seq = torch.cat([seq[:, 1:, :], x.unsqueeze(0).unsqueeze(0)], dim=1)
            p = model(seq) #Do the forward pass
            p = nn.functional.softmax(p, dim=0)
            move_robobo(p, rob)

def calc_loss(food_and_time: torch.Tensor, max_time: int, time_penalty: int, food_detect: FoodDetect):
    if food_and_time[0] > 0:
        n_food_reward = -torch.ceil(food_and_time[0]*(torch.log10(food_and_time[0]))) + 7
    else:
        n_food_reward = food_and_time[0] + 7
    sim_time = ((torch.pow(food_and_time[1], 2)*(1/max_time))) / max_time * time_penalty
    # c_food = food_detect.add_food(food_and_time[0])
    return n_food_reward + sim_time # + c_food

def train(rob, model: nn.Module, scaler: joblib.load, optimizer: torch.optim.Optimizer, max_time: int, time_penalty: int, seq: torch.Tensor) -> nn.Module:
    print('Started training')
    model.train()
    optimizer.zero_grad()
    for round_ in range(20): #Reset the robobo and target 10 times with or without random pos
        rob.play_simulation()
        start = time.time()
        food_detect = FoodDetect()
        for step in range(240): #Take max 75 steps per round
            robfn = RobFN.apply #Define the custom grad layer
            orientation = rob.read_orientation()
            accelleration = rob.read_accel()
            x = torch.tensor(scaler.transform([rob.read_irs()])[0].tolist() + [orientation.yaw, orientation.pitch, orientation.roll] + [accelleration.x, accelleration.y, accelleration.z], dtype=torch.float32)
            seq = torch.cat([seq[:, 1:, :], x.unsqueeze(0).unsqueeze(0)], dim=1)
            p = model(seq) #Do the forward pass
            p = p[0, -1, :]
            p = nn.functional.softmax(p, dim=0)
            food_and_time = robfn(p, *(rob, start)) #Calculate the custom robotics gradients

            loss = calc_loss(food_and_time, max_time, time_penalty, food_detect)

            print(f'round: {round_}, loss: {loss.item()}')
            loss.backward() #Do the backward pass
            optimizer.step() #Do a step in the learning space
            optimizer.zero_grad() #Clear the gradients in the optimizer
            if food_and_time[0] >= 7 or food_and_time[1] > 3*60*1000:
                print(f'object_completed within time: {food_and_time[1]}, collected: {food_and_time[0]}')
                break
        rob.stop_simulation()
        time.sleep(0.25)
    return model

def run_lstm_classification(
        rob: IRobobo, 
        max_time=3*60*1000, time_penalty=5, 
        seq_len=8, features=14, hidden_size=32, num_outputs=4, num_layers=2,
        eval_=False):
    # with torch.autograd.detect_anomaly():
    print('connected')
    # Setup things
    rob.set_phone_tilt_blocking(110, 100) #Angle phone forward
    scaler = joblib.load('software_powertrans_scaler.gz')

    seq = torch.zeros((1, seq_len, features), dtype=torch.float32)

    # Define the model and set it into train mode, together with the optimizer
    model = LSTM(features, hidden_size, num_outputs, num_layers)

    # Eval model in hw
    if not isinstance(rob, SimulationRobobo) or eval_:
        evaluation(rob, model, scaler, seq)
        return
    
    # Define optimizer for training
    optimizer = optim.Adam(params=model.parameters(), lr=0.05)
    model = train(rob, model, scaler, optimizer, max_time, time_penalty, seq)

    torch.save(model.state_dict(), './model.ckpt')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()