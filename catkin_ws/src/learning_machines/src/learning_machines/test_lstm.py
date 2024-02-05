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
import math
# import pandas as pd

def eucl_fn(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def get_img(rob: IRobobo, noise_thresh = 0, frame_name = 'frame', read_img = None):
    if read_img is None:
        img = rob.get_image_front()
    else:
        img = cv2.imread(read_img)

    img = cv2.resize(img, (640, 480))
    # Convert the image to HSV color space (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(str(f"./{frame_name}.png"), img)
    # cv2.imwrite(str("./frame_org.png"), img)

    # Define the color ranges for green, yellow, and white in HSV
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([80, 255, 255])

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_white = np.array([0, 0, 128])
    upper_white = np.array([255, 50, 255])

    lower_red = np.array([0, 100, 100])
    upper_red = np.array([10, 255, 255])

    # Create masks for each color
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv_image, lower_yellow, upper_yellow)
    mask_white = cv2.inRange(hsv_image, lower_white, upper_white)
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

    # Set the corresponding color values for each mask
    img[mask_green > 0] = [0, 255, 0]       # Green cubes
    img[mask_yellow > 0] = [0, 0, 0]      # Yellow floor
    img[mask_white > 0] = [255, 255, 255]   # White walls
    img[mask_red > 0] = [0, 0, 255]        # Red items
    
    img = img + (np.random.randn(img.shape[0], img.shape[1], img.shape[2]) * noise_thresh)

    cv2.imwrite(str(f"./frame.png"), img)

    # Create a mask for the green color
    mask_green = cv2.inRange(hsv_image, lower_green, upper_green)

    # Find contours in the mask
    contours_green, _ = cv2.findContours(mask_green, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image and calculate centroids for green items
    points_green = []
    for contour in contours_green:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points_green.append([cx, cy])

    # Create a mask for the red color
    mask_red = cv2.inRange(hsv_image, lower_red, upper_red)

    # Find contours in the mask
    contours_red, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image and calculate centroids for red items
    points_red = []
    for contour in contours_red:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            points_red.append([cx, cy])

    return img, points_green, points_red 

def move_robobo(movement, rob):
    movement = nn.functional.softmax(movement.detach(), dim=0)
    movement = movement.detach().numpy() #Use clone since detach doesnt properly work
    # move_pr = np.argmax(movement)
    move_pr = np.random.choice([0,1,2,3,4], p=movement)
    if move_pr == 0: #Move forward
        rob.move_blocking(50, 50, 250)
    elif move_pr == 1: #Move left
        rob.move_blocking(25, 50, 250)
    elif move_pr == 2: #Move right
        rob.move_blocking(50, 25, 250)
    elif move_pr == 3: #Rotate Left
        rob.move_blocking(-50, 50, 125)
    elif move_pr == 4: #Rotate Right
        rob.move_blocking(50, -50, 125)
    return move_pr

def evaluation(rob, model: nn.Module):
    model.load_state_dict(torch.load('./models/model_3.ckpt', map_location=torch.device('cpu')))
    print(sum(p.numel() for p in model.parameters() if p.requires_grad))
    model.eval()
    seq_length = 8
    with torch.no_grad():
        rob.set_phone_tilt_blocking(105, 100) #Angle phone forward
        seq = torch.zeros([1,seq_length,model.lstm_features])
        while True:
            # Get the input data
            img_, points_green, points_red = get_img(rob)
            x = torch.tensor(np.expand_dims(img_.swapaxes(-1, 0).swapaxes(-1, 1), 0), dtype=torch.float32)
            p, seq = model(x, seq) #Do the forward pass
            p = p[0]
            move_robobo(p, rob)

class Targetter():
    def __init__(self):
        self.previous_action = -1
        self.previous_green = False
        self.previous_red = False
    
    def get_target(self, points_green, points_red):
        target = -1
        width, height, center = 640, 480, 150
        red_box_in_place = False
        green_base_in_sight = len(points_green) > 0
        if len(points_red) > 0:
            for point in points_red: #Check if there is just a red box in front, otherwise ignore
                if point[0] > 640//2 - 30 and point[0] < 640//2 + 30 and \
                    point[1] > 400:
                    red_box_in_place = True
        if red_box_in_place and green_base_in_sight: #If we have the box and the green base is detected
            centroid, points_left, points_right = point_and_move(points_green, width, height, center)
            if centroid:
                target = 0
            elif points_left > points_right:
                target = 1
            else:
                target = 2
        elif len(points_red) > 0 and not red_box_in_place: #If we see the red box and dont have it yet
            centroid, points_left, points_right = point_and_move(points_red, width, height, center)
            if centroid:
                target = 0
            elif points_left > points_right:
                target = 1
            else:
                target = 2
        elif (self.previous_green and red_box_in_place) or (self.previous_red and not red_box_in_place):
            target = 4 if self.previous_action == 3 else 3
        elif self.previous_action == 4: #Otherwise keep going in the direction when rotating
            target = 4
        elif self.previous_action == 3:
            target = 3
        else: #This is more when the scene is init and we dont know anything
            if random.random() > 0.5:
                target = 3
            else:
                target = 4
        self.previous_action = target
        self.previous_green = green_base_in_sight
        self.previous_red = red_box_in_place
        return target, red_box_in_place, green_base_in_sight

def setup_rand(rob):
    # Set a random position of the robobo every round
    # rob.set_position(Position(-2.25-(random.random()*1.675), (random.random()*1.622)-0.048, 0.03971), Orientation(-90, -90, -90))
    # rob.set_food_position(Position(-3.925-(random.random()*1.675), (random.random()*1.622)-1.677, 0.03971))
    # rob top left: -2.25 , -0,048 , + 0.03971
    # rob bot right: -3.925 , 1.677 , + 0.03971      
    # food top left: -2.125 , -0.2 , +0.0111
    # food bot right: -4.125 , +1.8 ,  +0.0111
    # base: 1x1 and pos: -3.2 , +1,75 , +0.002 let food not spawn here
    # Constant for the allowed spawn point
    while True:
        rob.set_position(Position(-2.25-(random.random()*1.675), -0.048+(random.random()*1.725), 0.03971), Orientation(-90, -90, -90))
        rob.set_food_position(Position(-2.125-(random.random()*2), -0.2+(random.random()*2), 0.03971))
        
        # Check if the positions collide or are at the allowed spawn point
        robot_pos = rob.position()
        food_pos = rob.get_food_position()
        base_pos = rob.base_position()
        if not (food_pos.x > robot_pos.x - 0.1 and food_pos.x < robot_pos.x + 0.1 and \
                food_pos.y > robot_pos.y - 0.1 and food_pos.y < robot_pos.y + 0.1)  \
            and not(robot_pos.x > base_pos.x - 0.1 and robot_pos.x < base_pos.x + 0.1 and \
                robot_pos.y > base_pos.y - 0.1 and robot_pos.y < base_pos.y + 0.1) \
            and not(food_pos.x > base_pos.x - 0.1 and food_pos.x < base_pos.x + 0.1 and \
                food_pos.y > base_pos.y - 0.1 and food_pos.y < base_pos.y + 0.1):
            break

def train(rob, model: nn.Module, optimizer: torch.optim.Optimizer, max_steps=40, max_rounds=100, round_offset=0) -> nn.Module:
    print('Started training')
    model.train()
    optimizer.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    all_loss, all_actions, all_food, all_target = [], [], [], []
    seq_length = 64
    targetter = Targetter()
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: min(i / 500, 1.0))
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda i: max(-math.sqrt(i*(1/200))+2, 1.0))
    for round_ in range(max_rounds): #Reset the robobo and target 10 times with or without random pos
        round_ += round_offset
        rob.play_simulation()
        start = time.time()
        rob.set_phone_tilt_blocking(97, 100) #Angle phone forward
        loss_am, actions, food, target_am = [], [], [], []
        seq = torch.zeros([1,seq_length,model.lstm_features], requires_grad=True)
        early_resetting = False
        random_pos = False
        extra_steps = 0
        # 
        early_resetting = True
        steps_since = 0
        max_steps_since = 5
        if round_ > 5:
            max_steps_since = 10
        # elif round_ < 15 and round_ >= 6:
        #     setup_rand(rob)
        #     random_pos = True
        #     extra_steps = 100

        # rob.set_position(Position(-2.25-(random.random()*1.75), (random.random()*1.6)-0.048, 0.075), Orientation(-90, -90, -90))
        max_resets_overfitting_red = 100
        max_resets = 0
        for step in range(max_steps + extra_steps): # Keep going unless 3 minutes is reached or all food is collected
            did_optim = False
            img_, points_green, points_red = get_img(rob)
            x = torch.tensor(np.expand_dims(img_.swapaxes(-1, 0).swapaxes(-1, 1), 0), dtype=torch.float32)
            p, seq_new = model(x, seq) #Do the forward pass
            p = p[0]
            move = move_robobo(p, rob)
            target, red_box_in_place, green_box_in_place = targetter.get_target(points_green, points_red)

            target = torch.tensor(target, dtype=torch.long)
            loss = loss_fn(p, target)

            loss.backward() #Do the backward pass
            if step % 8 == 0 and step != 0:
                optimizer.step() #Do a step in the learning space
                optimizer.zero_grad() #Clear the gradients in the optimizer
                # scheduler.step() #Increase the learning rate
                did_optim = True
            seq = seq_new.detach()

            # if rob.nr_food_collected() > 0 and random_pos:
            #     print(f'Found red block in time: {time.time() - start}')
            #     break

            print(f'round: {round_}, loss: {loss.item()}, nr_food: {rob.nr_food_collected()}, target: {target.item()}, direction: {move}, greenbox in base: {green_box_in_place}, redbox in place: {red_box_in_place}')
            loss_am.append(str(loss.item()))
            actions.append(str(move))
            food.append(str(rob.nr_food_collected()))
            target_am.append(str(target.item()))
            if max_resets <= max_resets_overfitting_red and len(points_red) == 0 and early_resetting and not random_pos and steps_since >= max_steps_since:
                print('early resetting scene, too much wrong move')
                rob.stop_simulation()
                time.sleep(0.25)
                rob.play_simulation()
                rob.set_phone_tilt_blocking(97, 100) #Angle phone forward
                seq = torch.zeros([1,seq_length,model.lstm_features], requires_grad=True)
                steps_since = 0
                max_resets+=1
            elif max_resets <= max_resets_overfitting_red and len(points_red) == 0 and early_resetting:
                steps_since += 1
            else:
                steps_since = 0
            if rob.base_got_food() and not early_resetting and not random_pos:
                print(f'object_completed within time: {time.time() - start}')
                break
        if not did_optim:
            optimizer.step() #Do a step in the learning space
            optimizer.zero_grad() #Clear the gradients in the optimizer
            # scheduler.step() #Increase the learning rate
        all_loss.append(' '.join(loss_am))
        all_actions.append(' '.join(actions))
        all_food.append(' '.join(food))
        all_target.append(' '.join(target_am))
        with open(f'./res/res_loss_{round_}.txt', "w") as file_:
            file_.writelines(all_loss[-1])
        with open(f'./res/res_food_{round_}.txt', "w") as file_:
            file_.writelines(all_food[-1])
        with open(f'./res/res_target_{round_}.txt', "w") as file_:
            file_.writelines(all_target[-1])
        with open(f'./res/res_action_{round_}.txt', "w") as file_:
            file_.writelines(all_actions[-1])
        torch.save(model.state_dict(), f'./models/model_{round_}.ckpt')
        
        rob.stop_simulation()
        time.sleep(0.25)
    with open(f'./res/res_loss.txt', "w") as file_:
        file_.writelines(all_loss)
    with open(f'./res/res_food.txt', "w") as file_:
        file_.writelines(all_food)
    with open(f'./res/res_target.txt', "w") as file_:
        file_.writelines(all_target)
    with open(f'./res/res_action.txt', "w") as file_:
        file_.writelines(all_actions)
    return model

def point_and_move(points, width, height, center):
    points_right, points_left = 0, 0
    for point in points:
        # if eucl_fn(point, [640//2, 480//2]) < center: #If near center ignore the rest and continue forward
        if point[0] >= (width//2-center//2) and point[0] <= (width//2+center//2): #If near center ignore the rest and continue forward
            return True, None, None
        if point[0] <= (width//2-center//2): #If on the left
            points_left+=1
        else:
            points_right+=1
    return False, points_left, points_right

def run_lstm_classification(
        rob: IRobobo, 
        num_outputs=5, eval_=True):
    
    print('connected')
    if False:
        import glob, os
        round_offset = 0
        # Clean the dataset dir
        for file_ in glob.glob('./dataset/images/*.png'):
            os.remove(file_)
        for file_ in glob.glob('./dataset/*.csv'):
            os.remove(file_)
        round_ = 0 + round_offset
        while round_ < 200:
            rob.play_simulation()
            df = {
                'image': [],
                'target': [],
            }
            i = 0
            targetter = Targetter()
            setup_rand(rob)
            while True:
                rob.set_phone_tilt_blocking(97, 100) #Angle phone forward
                img, points_green, points_red = get_img(rob, frame_name=f'dataset/images/frame_{round_}_{i}')
                target = -1
                width, height, center = 640, 480, 150
                red_box_in_place = False
                target, red_box_in_place, green_box_in_place = targetter.get_target(points_green, points_red)
                df['image'].append(f'frame_{round_}_{i}.png')
                df['target'].append(target)
                if target == 0: #Move Forward
                    rob.move_blocking(50, 50, 500)
                elif target == 1: #Move left Forward
                    rob.move_blocking(25, 50, 250)
                elif target == 2: #Move right Forward
                    rob.move_blocking(50, 25, 250)
                elif target == 3: #Rotate Left
                    rob.move_blocking(-50, 50, 128)
                elif target == 4: #Rotate Right
                    rob.move_blocking(50, -50, 128)
                i+=1
                if rob.base_got_food() or i > 150:
                    if i > 150:
                        for file_ in glob.glob(f'./dataset/images/frame_{round_}_*.png'):
                            os.remove(file_)
                        for file_ in glob.glob(f'./dataset/{round_}.csv'):
                            os.remove(file_)
                        round_ -= 1
                    print(i)
                    round_ += 1
                    break
            pd.DataFrame.from_dict(df).to_csv(f'./dataset/{round_}.csv', index=False)
            rob.stop_simulation()
            time.sleep(0.25)
        return
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()
    # Define the model and set it into train mode, together with the optimizer
    # with torch.autograd.detect_anomaly():
    model = CNNwithLSTM(num_outputs)

    # Eval model in hw
    if not isinstance(rob, SimulationRobobo) or eval_:
        evaluation(rob, model)
        return
    
    # Define optimizer for training
    optimizer = optim.Adam(params=model.parameters(), lr=0.01)
    model = train(rob, model, optimizer)

    torch.save(model.state_dict(), './models/model.ckpt')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()