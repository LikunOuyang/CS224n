#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    """ CNN Model:
        - CNN Networks to combine these character embeddings
    """
    def __init__(self, 
                e_char, 
                f,
                k=5,
                max_word_len=21):
        """ Init CNN Model.

        @param e_char (int): Embedding size (dimensionality)
        @param k (int): Kernel Size (window size)
        @param f (int): number of filters
        """
        super(CNN, self).__init__()

        self.cnn = nn.Conv1d(e_char, f, kernel_size=k)
        self.max_pool = nn.MaxPool1d(max_word_len-k+1)


    def forward(self, x_reshaped):
        """ Take a mini-batch of character embedding lookup, compute the CNN output.

        @param emb (Tensor): Tensor of integers of shape (sentence_length, batch_size, e_char, max_word_length) where
            each integer is an element of the char embedding.

        @returns conv_out (Tensor): Tensor of integers of shape (sentence_length, batch_size, f)
        """

        x_conv = self.cnn(x_reshaped)
        conv_out = self.max_pool(nn.functional.relu(x_conv))
        return torch.squeeze(conv_out, -1)

### END YOUR CODE

