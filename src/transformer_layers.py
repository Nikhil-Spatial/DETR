from configs import D, H, W
from torch import sin, cos
import torch

def positional_encodings():
    pos_encodings = torch.empty(H, W, D)

    for pos_y in range(H):  # rows
        for pos_x in range(W):  # columns
            # compute y's half and x's half of positional encoding
            for i in range(0, D // 4):
                # sinusoidal positional encoding from "Attention Is All You
                # Need" paper
                scale = torch.tensor(10_000 ** (2 * i / (D / 2)))
                argument_y = pos_y / scale
                argument_x = pos_x / scale

                pos_encodings[pos_y][pos_x][2*i] = sin(argument_y)
                pos_encodings[pos_y][pos_x][2*i+1] = cos(argument_y)

                pos_encodings[pos_y][pos_x][(2*i)+128] = sin(argument_x)
                pos_encodings[pos_y][pos_x][(2*i+1)+128] = cos(argument_x)

    return pos_encodings.reshape(H*W, D)
