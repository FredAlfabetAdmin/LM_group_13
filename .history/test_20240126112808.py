import torch
weights = torch.load('./catkin_ws/model_best.ckpt')
print(weights)

weights = torch.load('./catkin_ws/model.ckpt')
print(weights)