from typing import List

import random
import torch
from torch import Tensor


class LoopProgram1(torch.jit.ScriptModule):
    def __init__(self, hidden_size, cells):
        super(LoopProgram1, self).__init__()
        self.cell1 = cells[0]
        self.cells = torch.nn.ModuleList(cells[1:])

        self.register_buffer('init_state', torch.zeros((1, hidden_size)))

    @torch.jit.script_method
    def forward(
            self,
            input_seq: List[List[Tensor]],  # Input data
            lens: List[int],
            batch_size: int,
            depth: int):
        """
        Argument 1 and Argument 2 are read-only input arrays.

        To make a program description independent of the size of data,
        parameters 3 to 5 are introduced which stand for possible size.
        """
        # `output_i` is the output buffer for loop i.
        output_i: List[List[List[Tensor]]] = []
        for i in range(batch_size):  # data parallelism
            output_d: List[List[Tensor]] = []
            output_d1: List[Tensor] = []

            prev = input_seq[i][0] + self.cell1(self.init_state)
            output_d1.append(prev)
            for t1 in range(1, lens[i], 1):
                # output_d1[t1] = ...
                y = input_seq[i][t1] + self.cell1(output_d1[t1 - 1])
                output_d1.append(y)
            output_d.append(output_d1)

            for d, cell in enumerate(self.cells):
                output_t: List[Tensor] = []

                prev = output_d[d][0] + self.cell1(self.init_state)
                output_t.append(prev)
                for t2 in range(1, lens[i], 1):
                    # output_t[t2] = ...
                    y = output_d[d][t2] + cell(output_t[t2 - 1])
                    output_t.append(y)
                output_d.append(output_t)
            output_i.append(output_d)
        return output_i


def run_test(seq_batch,
             lens,
             batch_size,
             depth,
             input_size,
             hidden_size,
             device='cuda'):

    cells = [torch.nn.Linear(hidden_size, hidden_size) for _ in range(depth)]
    m = LoopProgram1(hidden_size, cells).to(device)
    print(torch.jit.script(m).graph)

    m(seq_batch, lens, batch_size, depth)


if __name__ == '__main__':
    batch_size = 13
    depth = 5
    min_len = 5
    max_len = 15

    input_size = 32
    hidden_size = input_size

    device = 'cuda'

    random.seed(1234)
    torch.manual_seed(1234)

    seq_batch = []
    lens = [random.randint(min_len, max_len) for _ in range(batch_size)]
    for i in range(batch_size):
        seq_batch.append([])
        for j in range(lens[i]):
            seq_batch[-1].append(torch.randn(1, input_size, device=device))

    run_test(seq_batch, lens, batch_size, depth, input_size, hidden_size,
             device)
