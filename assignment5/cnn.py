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
    def __init__(self, e_char, f):
        """ Init CNN Model.

        @param e_char (int): Embedding size (dimensionality)
        @param k (int): Kernel Size (window size)
        @param f (int): number of filters
        """
        super(CNN, self).__init__()

        self.cnn = nn.Conv1d(e_char, f, kernel_size=5, padding=1)
        self.max_pool = nn.MaxPool1d(21-5+1)


    def forward(self, emb):
        """ Take a mini-batch of character embedding lookup, compute the CNN output.

        @param emb (Tensor): Tensor of integers of shape (sentence_length, batch_size, e_char, max_word_length) where
            each integer is an element of the char embedding.

        @returns conv_out (Tensor): Tensor of integers of shape (sentence_length, batch_size, f)
        """
        sentence_length, batch_size = emb.shape[0], emb.shape[1]
        x_conv = self.cnn(emb.view(-1, emb.shape[2], emb.shape[3]))
        conv_out = self.max_pool(nn.functional.relu(x_conv))
        return conv_out.view(sentence_length, batch_size, -1)

### END YOUR CODE

