import torch
import torch.nn as nn
import torch.nn.functional as F

#A class for creating residual blocks consisting of two convolutional layers:
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        #The first 3 x 3 convolution:
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        #To normalize the output of the first convolution:
        self.bn1 = nn.BatchNorm2d(channels)
        #The second 3 x 3 convolution:
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        #To normalize the output of the second convolution:
        self.bn2 = nn.BatchNorm2d(channels)

    #A residual connection is created by adding the input to the output:
    def forward(self, x):
        residual = x
        #Applying the convolutions:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

#The following is a class for creating a residual network.  The class takes the number of channels in the state observation, the number of residual blocks, and the board size as input and returns three Q-values and a state value:
class ResidualNetwork(nn.Module):
    def __init__(self, input_channels=10, num_res_blocks=10, board_size=11):
        super().__init__()
        self.board_size = board_size
        #Converting the observation channels into 128 feature maps:
        self.initial_conv = nn.Conv2d(input_channels, 128, kernel_size=3, padding=1)
        #Normalizing the output of the initial convolution:
        self.bn_initial = nn.BatchNorm2d(128)
        #Creating a sequence of residual blocks:
        self.res_blocks = nn.Sequential(*[ResBlock(128) for _ in range(num_res_blocks)])

        #The advantage/policy head of the network:
        self.policy_head = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=1),
            nn.BatchNorm2d(64),
            nn.Flatten(),
            #A fully connected layer:
            nn.Linear(64 * board_size * board_size, 256),
            nn.ReLU(),
            #To compute an advantage value for each of the three actions:
            nn.Linear(256, 3)
        )

        #The value head of the network:
        self.value_head = nn.Sequential(
            nn.Conv2d(128, 32, kernel_size=1), 
            nn.BatchNorm2d(32),
            nn.Flatten(),
            #A fully connected layer:
            nn.Linear(32 * board_size * board_size, 256),
            nn.ReLU(),
            #To compute the estimated value of the current state:
            nn.Linear(256, 1),
            nn.Tanh()
        )

        #Initializing the output layers of the network with small weights and biases to reduce the value of the initial outputs:
        nn.init.uniform_(self.policy_head[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.value_head[-2].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.policy_head[-1].bias, -3e-3, 3e-3)
        nn.init.uniform_(self.value_head[-2].bias, -3e-3, 3e-3)

    def forward(self, x):
        #Passing the observation through the initial convolution and residual blocks to extract features:
        x = F.relu(self.bn_initial(self.initial_conv(x)))
        x = self.res_blocks(x)
        
        #To compute the state value and three advantage values:
        v = self.value_head(x)
        a = self.policy_head(x)

        #Substituting the state and advantage values into the dueling network formula:
        q = v + (a - a.mean(dim=1, keepdim=True))
        #Returning 3 Q-values and the estimated state value:
        return q, v


