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

def eucl_fn(point1, point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def get_img(rob: IRobobo, noise_thresh = 0):
    img = rob.get_image_front()

    img = cv2.resize(img, (640, 480))
    # Convert the image to HSV color space (Hue, Saturation, Value)
    hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imwrite(str("./frame_org.png"), img)

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

    cv2.imwrite(str("./frame.png"), img)

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
    move_pr = np.random.choice([0,1,2,3], p=movement)
    if move_pr == 0: #Move forward
        rob.move_blocking(50, 50, 250)
    elif move_pr == 1: #Move backward
        rob.move_blocking(-25, -60, 250)
    elif move_pr == 2: #Move left
        rob.move_blocking(25, 50, 250)
    elif move_pr == 3: #Move right
        rob.move_blocking(50, 25, 250)
    return move_pr

def evaluation(rob, model: nn.Module):
    model.load_state_dict(torch.load('./model_49.ckpt'))
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

def get_target(points_green, points_red):
    target = -1
    width, height, center = 640, 480, 150
    red_box_in_place = False
    if len(points_red) > 0:
        for point in points_red: #Check if there is just a red box in front, otherwise ignore
            if point[0] > 640//2 - 30 and point[0] < 640//2 + 30 and \
                point[1] > 415:
                red_box_in_place = True
    if red_box_in_place and len(points_green) > 0:
        centroid, points_left, points_right = point_and_move(points_green, width, height, center)
        if centroid:
            target = 0
        elif points_left >= points_right:
            target = 2
        else:
            target = 3
    elif len(points_red) > 0 and not red_box_in_place:
        centroid, points_left, points_right = point_and_move(points_red, width, height, center)
        if centroid:
            target = 0
        elif points_left >= points_right:
            target = 2
        else:
            target = 3
    # elif red_box_in_place:
    #     rnd = random.random()
    #     target = 3 if rnd > 0.5 else 2
    else:
        target = 1
    return target, red_box_in_place

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
            and not (robot_pos.x > base_pos.x - 0.1 and robot_pos.x < base_pos.x + 0.1 and \
                robot_pos.y > base_pos.y - 0.1 and robot_pos.y < base_pos.y + 0.1) \
            and not (food_pos.x > base_pos.x - 0.1 and food_pos.x < base_pos.x + 0.1 and \
                food_pos.y > base_pos.y - 0.1 and food_pos.y < base_pos.y + 0.1):
            break

def train(rob, model: nn.Module, optimizer: torch.optim.Optimizer, max_steps=100, max_rounds=50, round_offset=0) -> nn.Module:
    print('Started training')
    model.train()
    optimizer.zero_grad()
    loss_fn = nn.CrossEntropyLoss()
    all_loss, all_actions, all_food, all_target = [], [], [], []
    seq_length = 8
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
        if round_ < 1:
            early_resetting = True
        # elif round_ < 15 and round_ >= 5:
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
            target, red_box_in_place = get_target(points_green, points_red)

            target = torch.tensor(target, dtype=torch.long)
            loss = loss_fn(p, target)

            loss.backward() #Do the backward pass
            if step % 16 == 0 and step != 0:
                optimizer.step() #Do a step in the learning space
                optimizer.zero_grad() #Clear the gradients in the optimizer
                # scheduler.step() #Increase the learning rate
                did_optim = True
            seq = seq_new.detach()

            # if rob.nr_food_collected() > 0 and random_pos:
            #     print(f'Found red block in time: {time.time() - start}')
            #     break

            print(f'round: {round_}, loss: {loss.item()}, nr_food: {rob.nr_food_collected()}, target: {target.item()}, direction: {move}, learning_rate: {optimizer.param_groups[0]["lr"]}, redbox in place: {red_box_in_place}')
            loss_am.append(str(loss.item()))
            actions.append(str(move))
            food.append(str(rob.nr_food_collected()))
            target_am.append(str(target.item()))
            if max_resets <= max_resets_overfitting_red and len(points_red) == 0 and early_resetting and not random_pos:
                print('early resetting scene, too much wrong move')
                rob.stop_simulation()
                time.sleep(0.25)
                rob.play_simulation()
                rob.set_phone_tilt_blocking(97, 100) #Angle phone forward
                seq = torch.zeros([1,seq_length,model.lstm_features], requires_grad=True)
                max_resets+=1
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

def point_and_move(points, width, height, center):
    points_right, points_left = 0, 0
    for point in points:
        # if eucl_fn(point, [640//2, 480//2]) < center: #If near center ignore the rest and continue forward
        if point[0] >= (width//2-center//2) and point[0] <= (width//2+center//2): #If near center ignore the rest and continue forward
            target = 0
            return True, None, None
        if point[0] <= (width//2-center//2): #If on the left
            points_left+=1
        else:
            points_right+=1
    return False, points_left, points_right

def run_lstm_classification(
        rob: IRobobo, 
        num_outputs=4, eval_=False):
    
    if isinstance(rob, SimulationRobobo):
        rob.play_simulation()

    print('connected')
    if False:
        while True:
            rob.set_phone_tilt_blocking(97, 100) #Angle phone forward
            img, points_green, points_red = get_img(rob)
            target = -1
            width, height, center = 640, 480, 150
            red_box_in_place = False
            if len(points_red) > 0:
                for point in points_red: #Check if there is just a red box in front, otherwise ignore
                    if point[0] > 640//2 - 30 and point[0] < 640//2 + 30 and \
                        point[1] > 415:
                        red_box_in_place = True
            print(red_box_in_place, points_red)
            if red_box_in_place and len(points_green) > 0:
                centroid, points_left, points_right = point_and_move(points_green, width, height, center)
                if centroid:
                    target = 0
                    rob.move_blocking(50,50,500)
                elif points_left >= points_right:
                    target = 2
                    rob.move_blocking(25,50,250)
                else:
                    target = 3
                    rob.move_blocking(50,25,250)
            elif len(points_red) > 0 and not red_box_in_place:
                centroid, points_left, points_right = point_and_move(points_red, width, height, center)
                if centroid:
                    target = 0
                    rob.move_blocking(50,50,500)
                elif points_left >= points_right:
                    target = 2
                    rob.move_blocking(25,50,250)
                else:
                    target = 3
                    rob.move_blocking(50,25,250)
            else:
                target = 1
                rob.move_blocking(-25, -60, 250)
            time.sleep(1)
        return
    # Define the model and set it into train mode, together with the optimizer
    # with torch.autograd.detect_anomaly():
    model = CNNwithLSTM(num_outputs)

    # Eval model in hw
    if not isinstance(rob, SimulationRobobo) or eval_:
        evaluation(rob, model)
        return
    
    # Define optimizer for training
    optimizer = optim.Adam(params=model.parameters(), lr=0.005)
    model = train(rob, model, optimizer)

    torch.save(model.state_dict(), './model.ckpt')

    if isinstance(rob, SimulationRobobo):
        rob.stop_simulation()