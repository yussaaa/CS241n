#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""

### YOUR CODE HERE for part 1i
import torch, torch.nn as nn, torch.nn.functional as F

class CNN(nn.Module):
    """
    Implementing a 1-d CNN with
    ReLu activation layer
    Max-pooling layer

    Dimension Notations:
        b: Batch size
        e_char: Character embedding size 50
        e_word: Word embedding size 256
        max_word: Maximum word length 21
        sen_len: Maximum sentence length: max_sent_length
    """

    def __init__(self, input_channels, output_channels, kernel_size=5):
        """
        Initialize the 1-d CNN layer

        Use the adaptive max-pooling because we don't need to select the
        hyper-parameter like kernel size, stride etc.
        Args:
            input_channels (int): Input channels/depth
            output_channels (int): Output channels/depth, in this case will the
                                    embed_word_size (e_word)
            kernel_size (int): Size of the filter to extract info from input,
                                Default: 5
        """
        super(CNN, self).__init__()
        self.conv = nn.Conv1d(in_channels=input_channels,
                              out_channels=output_channels,
                              kernel_size=kernel_size,
                              )
        self.max_pool = nn.AdaptiveMaxPool1d(output_size=1)

    def forward(self, input):
        """
        Define the forward pass of the Char-CNN from x_reshaped to x_conv_out

        Args:
            input (torch.tensor): Input reshaped tensor.
                                    with shape(b*sen_len, e_char, e_word)

        Returns:
            x_conv_out (torch.tensor): Output tensor
                                    shape: (b*sen_len, e_word)
        """
        x_conv = self.conv(input) # (e_word, m_word-k+1)
        activation = F.relu(x_conv)
        x_conv_out = self.max_pool(activation).squeeze(dim=2)
        return x_conv_out
### END YOUR CODE
