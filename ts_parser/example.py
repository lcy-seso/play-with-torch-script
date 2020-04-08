from typing import List, Tuple
import random

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter
from torch.nn.init import xavier_normal_, zeros_


class MyCell(nn.Module):
    """Simulate user-defined cells."""

    def __init__(self, input_size, hidden_size):
        super(MyCell, self).__init__()
        self.W = Parameter(Tensor(input_size, hidden_size))

        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                xavier_normal_(p.data)
            else:
                zeros_(p.data)

    def forward(self, input: Tensor) -> Tensor:
        return torch.mm(input, self.W)


class MyModule(nn.Module):
    def __init__(self, cell):
        super(MyModule, self).__init__()

        self.cell = cell

    def forward(self, input: List[Tensor], batch_size: int) -> List[Tensor]:

        output: List[Tensor] = []
        for i in range(batch_size):
            x = self.cell(input[i])
            output.append(x)
        return output


def forward(self, input: List[Tensor], batch_size: int, cell) -> List[Tensor]:

    output: List[Tensor] = []
    for i in range(batch_size):
        x = cell(input[i])
        output.append(x)
    return output


if __name__ == '__main__':
    random.seed(5)
    torch.manual_seed(5)

    device = 'cpu'

    batch_size = 4

    input_size = 16
    hidden_size = 16

    a_seq = [
        torch.randn(1, input_size, device=device) for _ in range(batch_size)
    ]

    outputs = []

    cell = MyCell(input_size, hidden_size)
    m = MyModule(cell).to(device)

    ts = torch.jit.get_jit_def(forward)
    print(ts)

    # ts = torch.jit.script(m)
    # print(ts.graph)

    m(a_seq, batch_size)
