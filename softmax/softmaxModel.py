import torch.nn as nn


class SoftmaxModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10, bias=True).to("cuda")

    def forward(self, x):
        return self.linear(x)
