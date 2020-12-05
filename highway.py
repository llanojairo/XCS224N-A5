#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1d
# Implement the highway network as a nn.Module class called Highway

import torch
import torch.nn as nn
import torch.nn.functional as F

# Highway network for Conv ENC
class Highway(nn.Module):
    def __init__(self, size_embedding):
        """ 
        Initialize Highway Net
        @param size_embedding: word embedding size
        """
        super(Highway, self).__init__()
        self.projection = nn.Linear(size_embedding, size_embedding)
        self.gate = nn.Linear(size_embedding, size_embedding)

    def forward(self, conv_out_X: torch.Tensor) -> torch.Tensor:
        """
            Forward pass for batch of sentences
            @param conv_out_X: tensor with shape [max_sentence_length, batch_size, size_embedding]
            @return highway_X: combined output with shape [max_sentence_length, batch_size, size_embedding]
        """
        projection_X = F.relu(self.projection(conv_out_X))
        gate_X = torch.sigmoid(self.gate(conv_out_X))
        highway_X = torch.mul(projection_X, gate_X) + torch.mul(conv_out_X, 1 - gate_X)

        return highway_X

### END YOUR CODE 

