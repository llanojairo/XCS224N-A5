#!/usr/bin/env python3
# -*- coding: utf-8 -*-

### YOUR CODE HERE for part 1e
import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
	"""
	Initialize 1-D CNN.
	@param embed_size: embedding size of characters
	@param window_size: window size
	@param filters_num: # of filters (embedding size)
	"""
	def __init__(self, embed_size: int = 50, m_word: int = 21, window_size: int = 5, filters_num: int = None):
		super(CNN, self).__init__()
		self.conv1d = nn.Conv1d(in_channels=embed_size, out_channels=filters_num, kernel_size=window_size)
		self.maxpool = nn.MaxPool1d(kernel_size=m_word - window_size + 1)

	def forward(self, reshaped_X: torch.Tensor) -> torch.Tensor:
		"""
		reshaped_X to conv_out_X
		@param reshaped_X: char-level embedding tensor
		@return conv_out_X: word-level embedding tensor
		"""
		conv_X = self.conv1d(reshaped_X)
		conv_out_X = self.maxpool(F.relu(conv_X))

		return torch.squeeze(conv_out_X, -1)

### END YOUR CODE

