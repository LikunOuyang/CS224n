#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch
import torch.nn as nn
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    """ Highway Model:
        - Highway Networks6 have a skip-connection controlled by a dynamic gate
    """
    def __init__(self, embed_size, dropout_rate=0.2):
        """ Init Highway Model.

        @param embed_size (int): Embedding size (dimensionality)
        @param dropout_rate (float): Dropout probability, for attention
        """
        super(Highway, self).__init__()

        self.W_proj = nn.Linear(embed_size, embed_size)
        self.W_gate = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout3d(dropout_rate)

    def forward(self, conv_out):
        """ Take a mini-batch of output from CNN, compute the word embedding.

        @param conv_out (Tensor): Tensor of integers of shape (sentence_length, batch_size, embed_size)

        @returns word_emb (Tensor): Tensor of integers of shape (sentence_length, batch_size, embed_size)
        """
        
        x_proj = nn.functional.relu(self.W_proj(conv_out))
        x_gate = torch.sigmoid(self.W_gate(conv_out))
        x_highway = x_gate*x_proj + (1-x_gate)*conv_out
        word_emb = self.dropout(x_highway)
        return word_emb

### END YOUR CODE 

