import os
import inspect
import sys
import ast
import pprint as pp

sys.path.insert(0,
                os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List
import random
import torch
from torch import Tensor

from utils import VanillaRNNCell


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
    output_i: List[List[List[Tensor]]] = []
    for i in range(batch_size):  # parallelizable loop

        output_j: List[List[Tensor]] = []
        for d, cell in enumerate(self.cells):

            output_k: List[Tensor] = []
            for t in range(lens[i]):  # iterate over variable length data
                if d == 0:
                    x = input_seq[i][t]
                else:
                    x = output_j[d - 1][t]

                if t == 0:
                    h_prev = self.init_state
                else:
                    h_prev = output_k[-1]

                h = cell(x, h_prev)
                output_k.append(h)
            output_j.append(output_k)
        output_i.append(output_j)
    return output_i


def dump_python_ast():
    sourcelines, file_lineno = inspect.getsourcelines(forward)
    root = ast.parse(''.join(sourcelines))
    pp.pprint(ast.dump(root))


def dump_ts_ast():
    print(torch.jit.get_jit_def(forward))


if __name__ == '__main__':
    # dump_python_ast()
    dump_ts_ast()
