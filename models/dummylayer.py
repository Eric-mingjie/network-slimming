import torch
import torch.nn as nn

class dummy_layer(nn.Module):
	def __init__(self):
		super(dummy_layer, self).__init__()

	def forward(self, input_tensor):
		return input_tensor