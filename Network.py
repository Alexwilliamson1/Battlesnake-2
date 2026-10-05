import torch
import torch.nn as nn
import torch.nn.functional as F

# A convolutional dueling Q-network for an 11 x 11 Battlesnake board.
# The network takes 10 observation channels as input and produces
# three Q-values corresponding to left, straight, and right.
class ConvolutionalNeuralNetwork(nn.Module):

    def __init__(
        self,
        input_channels=10,
        board_size=11
    ):
        super().__init__()

        self.board_size = board_size

        # Feature extraction layers.
        # Padding keeps the spatial dimensions at 11 x 11.
        self.features = nn.Sequential(

            # 10 x 11 x 11 -> 128 x 11 x 11
            nn.Conv2d(
                input_channels,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),

            # 128 x 11 x 11 -> 128 x 11 x 11
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),

            # 128 x 11 x 11 -> 128 x 11 x 11
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(16, 128),
            nn.ReLU(inplace=True),

            # 128 x 11 x 11 -> 256 x 11 x 11
            nn.Conv2d(
                128,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            # 256 x 11 x 11 -> 256 x 11 x 11
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),

            # 256 x 11 x 11 -> 256 x 11 x 11
            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True)
        )

        # Shared fully connected representation.
        self.shared = nn.Sequential(
            nn.Flatten(),

            nn.Linear(
                256 * board_size * board_size,
                512
            ),

            nn.ReLU(inplace=True)
        )

        # Advantage head:
        # produces one advantage value for each action:
        # left, straight, and right.
        self.advantage_head = nn.Linear(
            512,
            3
        )

        # Value head:
        # estimates the value of the current state.
        self.value_head = nn.Linear(
            512,
            1
        )

        self._initialize_weights()

    def _initialize_weights(self):

        # He/Kaiming initialization is appropriate for ReLU networks.
        for module in self.modules():

            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu"
                )

            elif isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(
                    module.weight,
                    nonlinearity="relu"
                )

                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Keep the final Q-related outputs initially small.
        nn.init.uniform_(
            self.advantage_head.weight,
            -3e-3,
            3e-3
        )
        nn.init.uniform_(
            self.advantage_head.bias,
            -3e-3,
            3e-3
        )

        nn.init.uniform_(
            self.value_head.weight,
            -3e-3,
            3e-3
        )
        nn.init.uniform_(
            self.value_head.bias,
            -3e-3,
            3e-3
        )

    def forward(self, x):

        # Extract spatial features from the board.
        x = self.features(x)

        # Convert the convolutional feature maps into a shared
        # representation for the value and advantage heads.
        x = self.shared(x)

        # Compute the three action advantages.
        a = self.advantage_head(x)

        # Compute the state value.
        v = self.value_head(x)

        # Dueling-network aggregation:
        #
        # Q(s, a) = V(s) +
        #           A(s, a) -
        #           mean(A(s, ·))
        q = v + (
            a - a.mean(
                dim=1,
                keepdim=True
            )
        )

        # q contains three values:
        # q[:, 0] = left
        # q[:, 1] = straight
        # q[:, 2] = right
        return q, v
