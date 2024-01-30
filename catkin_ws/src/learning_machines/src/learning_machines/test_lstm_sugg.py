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

from .lstm import *
from torch import optim, nn
import torch, time, random
import numpy as np
import joblib
import cv2
import os
import json
''' 
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

    def get_grey(self, rob: IRobobo):
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

        return gray_frame, frame

    def blob_detect(self, rob: IRobobo):
        gray_frame, frame = self.get_grey(rob)

        #perform blob detection
        keypoints = self.detector.detect(gray_frame)

        cv2.imwrite(str("./frame.png"), frame)
        cv2.imwrite(str("./gray_frame.png"), gray_frame)
        
        x, y = 0, 0.5
        size_percent = gray_frame[gray_frame > 0].shape[0] / (gray_frame.shape[0] * gray_frame.shape[1]) * 100

        if keypoints:
            keypoint = keypoints[0]
            x, y = int(keypoint.pt[0]) / self.camera_width, int(keypoint.pt[1]) / self.camera_height
            # size_percent = (keypoint.size / (self.camera_width * self.camera_height)) * 100
            #x and y values along with the percentage of blob area
        return [x, y, size_percent]
        
def move_robobo(movement, rob):
    movement = nn.functional.softmax(movement.detach(), dim=0)
    movement = movement.detach().numpy() #Use clone since detach doesnt properly work
    move_pr = np.argmax(movement)
    if move_pr == 0: #Move forward
        movement = [50, 50, 250]
    elif move_pr == 1: #Move backward
        movement = [-50, -50, 250]
    elif move_pr == 2: #Move left
        movement = [-50, 50, 125]
    elif move_pr == 3: #Move right
        movement = [50, -50, 125]
    rob.move_blocking(int(movement[0]), int(movement[1]), int(movement[2]))
    return move_pr

def evaluation(rob, model: nn.Module, detector: Blob_Detection):
    model.load_state_dict(torch.load('./model.ckpt'))
    model.eval()
    with torch.no_grad():
        rob.set_phone_tilt_blocking(105, 100) #Angle phone forward
        while True:
            # Get the input data
            img_, _ = detector.get_grey(rob)
            p = model(img_) #Do the forward pass
            detector.blob_detect(rob)
            move_robobo(p, rob)

def train(rob, model: nn.Module, optimizer: torch.optim.Optimizer, detector: Blob_Detection) -> nn.Module:
    print('Started training')
    model.train()
    optimizer.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    all_loss, all_actions, all_food, all_target = [], [], [], []
    for round_ in range(50): #Reset the robobo and target 10 times with or without random pos
        rob.play_simulation()
        start = time.time()
        rob.set_phone_tilt_blocking(105, 100) #Angle phone forward
        loss_am, actions, food, target_am = [], [], [], []
        for _ in range(20):
            # We train a visual classifier in the environment.
            img_, _ = detector.get_grey(rob)
            rob_position = rob.position()
            x = torch.tensor([img_], dtype=torch.float32)
            p = model(x) #Do the forward pass
            p = p[0]
            # p = 
            move = move_robobo(p, rob)
            new_position = rob.position()
            amount_green = detector.blob_detect(rob)[-1]
            if amount_green > 0: #If we see any green, continue forward
                # Wall check otherwise keep going
                if eucl_loss_fn(torch.tensor([rob_position.x, rob_position.y], dtype=torch.float32), torch.tensor([new_position.x, new_position.y], dtype=torch.float32)) < 0.05:
                    target = 1
                else:
                    target = 0
            # Otherwise we are in the wall most likely then:
            elif eucl_loss_fn(torch.tensor([rob_position.x, rob_position.y], dtype=torch.float32), torch.tensor([new_position.x, new_position.y], dtype=torch.float32)) < 0.05:
                target = 1
            # If we are doing that, but there are not enough green than turn.
            else:
                if np.random.rand() > 0.5:
                    target = 2
                else:
                    target = 3
            target = torch.tensor(target, dtype=torch.long)
            loss = loss_fn(p, target)

            loss.backward() #Do the backward pass
            optimizer.step() #Do a step in the learning space
            optimizer.zero_grad() #Clear the gradients in the optimizer
            print(f'round: {round_}, loss: {loss.item()}, nr_food: {rob.nr_food_collected()}, target: {target.item()}, direction: {move}')
            loss_am.append(str(loss.item()))
            actions.append(str(move))
            food.append(str(rob.nr_food_collected()))
            target_am.append(str(target.item()))
            if rob.nr_food_collected() >= 7 or time.time() - start > 3*60*1000:
                print(f'object_completed within time: {time.time() - start}, collected: {rob.nr_food_collected()}')
                break
        all_loss.append(' '.join(loss_am))
        all_actions.append(' '.join(actions))
        all_food.append(' '.join(food))
        all_target.append(' '.join(target_am))
        with open(f'./res_loss_{round_}.txt', "w") as file_:
            file_.writelines(all_loss[-1])
        with open(f'./res_food_{round_}.txt', "w") as file_:
            file_.writelines(all_food[-1])
        with open(f'./res_target_{round_}.txt', "w") as file_:
            file_.writelines(all_target[-1])
        with open(f'./res_action_{round_}.txt', "w") as file_:
            file_.writelines(all_actions[-1])
        torch.save(model.state_dict(), f'./model_{round_}.ckpt')
        
        rob.stop_simulation()
        time.sleep(0.25)
    with open(f'./res_loss.txt', "w") as file_:
        file_.writelines(all_loss)
    with open(f'./res_food.txt', "w") as file_:
        file_.writelines(all_food)
    with open(f'./res_target.txt', "w") as file_:
        file_.writelines(all_target)
    with open(f'./res_action.txt', "w") as file_:
        file_.writelines(all_actions)
    return model
'''
def run_lstm_classification(rob: IRobobo):
    # Place the Robobo in a random position
    # Place the food blob in a random position
    # Take a screenshot from the camera
    # Save the two positions into a file.
    # Make sure to connect the ID's.
    directory = './training_data/'
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    size_training_data = 3
    for x in range(size_training_data):
        # Randomize the positions of the robobo and the food.


        # Gather the information
        wheel_position = rob.read_wheels()
        view = rob.get_image_front()
        food_position = rob.get_food_position()
        wheel_orientation = rob.read_orientation()

    
        scenario = {
            "ID":str(x),
            "wheel_position": wheel_position,
            "wheel_orientation":wheel_orientation,
            "food_position":food_position
        }

        
        # Write info
        with open(f"{directory}/train_{x}.json", "w") as file:
            json.dump(scenario, file, indent=2)
        
        # Save image
        cv2.imwrite(str("{directory}/train_{x}.json.png"), view)