import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pygame as pg

pg.init()

screen = pg.display.set_mode((500, 500))

device = torch.device('cpu')
if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


np.random.seed(5)

input_height = input_width = 250
class conv_network(nn.Module):
    def __init__(self, input_size, output_size, lr=0.0025) -> None:
        super(conv_network, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool_1 = nn.MaxPool2d(1, 1)
        self.conv2 = nn.Conv2d(6, 16, 3)

        self.fcl_1 = nn.Linear(16 * 1 * 1, 64)
        self.fc_2 = nn.Linear(64, output_size)
        self.fc_3 = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.pool_1(F.relu(self.conv1(x)))
        x = self.pool_1(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 1 * 1)
        x = F.relu(self.fcl_1(x))
        x = self.fc_3(self.fc_2(x))

        return x

class conv_model(nn.Module):
    def __init__(self) -> None:
        super(conv_model, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, 3)
        self.pool_1 = nn.MaxPool2d(1, 1)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.pool_2 = nn.MaxPool2d(1, 1)

        self.fcl_1 = nn.Linear(16 * 1 * 1, 64)
        self.fc_2 = nn.Linear(64, 1)
        self.fc_3 = nn.Linear(1, 1)

    def forward(self, x):
        x = self.pool_1(F.relu(self.conv1(x)))        
        x = self.pool_2(F.relu(self.conv2(x)))

        x = x.view(-1, 16 * 1 * 1)
        x = F.relu(self.fcl_1(x))
        x = self.fc_3(self.fc_2(x))

        return x

image_state = np.random.randn(1, 1, 5, 5)
state_tensor = torch.tensor(image_state, dtype=torch.float32).to(device)

criterion = nn.MSELoss()
# model = conv_model()
model = conv_network(5, 10)
model.to(device)
label = np.ones((4, 1))
label_tensor = torch.tensor(label, dtype=torch.float32).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for _ in range(100) :
    for event in pg.event.get() :
        if event.type == pg.QUIT :
            break

    screen.fill((255, 200, 150))
    pg.display.flip()

    output = model(state_tensor)

    loss = criterion(output, label_tensor)

    print(loss)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




# print(output)