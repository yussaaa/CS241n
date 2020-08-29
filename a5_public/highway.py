#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1h
import torch
import torch.nn as nn
import torch.nn.functional as F

class Highway(nn.Module):
    """
    Highway network transferring x_conv --> x_highway
    x_conv.shape:(max_sentence_length, batch_size, max_word_length)

    Middle step:
        x_proj, x_gate
    """
    def __init__(self, input_size):
        """
        Initiate the Highway network.

        :param size (int): Size of the input tensor
            Note: the input & output tensor size are same
        """
        super(Highway, self).__init__()
        self.proj = nn.Linear(input_size, input_size)
        self.gate = nn.Linear(input_size, input_size)

    def forward(self, x_input):
        """
        Forward pass of the highway network
        :arg: x_input (torch.tensor): the input tensor

        :return: x_highway (torch.tensor): The return tensor
        """
        proj = F.relu(self.proj(x_input))
        gate = torch.sigmoid(self.gate(x_input))
        highway  = gate * proj + (1-gate) * x_input

        return highway
### END YOUR CODE

