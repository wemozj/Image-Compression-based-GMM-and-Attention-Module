"""
GCAM : Global Context Attention Module
"""
from .context_block import ContextBlock
import torch.nn as nn
import torch
import math

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)

# 1x1 convolution
def conv1x1(in_channels, out_channels, stride=1):
	return nn.Conv2d(in_channels, out_channels, kernel_size=1,
	                 stride=stride, padding=0, bias=False)

# Residual block
class ResidualBlock(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1, downsample=None):
		super(ResidualBlock, self).__init__()
		self.conv1 = conv3x3(in_channels, out_channels)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = conv3x3(out_channels, out_channels)
		self.bn2 = nn.BatchNorm2d(out_channels)
		self.downsample = downsample
		
		# init weight
		torch.nn.init.xavier_normal_(self.conv1.weight.data, math.sqrt(2))
		# torch.nn.init.constant_(self.conv1.bias.data, 0.01)
		torch.nn.init.xavier_normal_(self.conv2.weight.data, math.sqrt(2))
		# torch.nn.init.constant_(self.conv2.bias.data, 0.01)

	
	
	def forward(self, x):
		residual = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)
		if self.downsample:
			residual = self.downsample(x)
		out += residual
		# out = self.relu(out)
		return out

class GCAM(nn.Module):
	def __init__(self, in_channels, out_channels):
		super(GCAM, self).__init__()
		self.GCBlock = ContextBlock(in_channels, ratio=out_channels/in_channels)
		# self.ResBlock = ResidualBlock(in_channels, out_channels)
		# self.conv1x1 = conv1x1(in_channels, out_channels)
		
		# init weight
		# torch.nn.init.xavier_normal_(self.conv1x1.weight.data, math.sqrt(2))
		# torch.nn.init.constant_(self.conv1x1.bias.data, 0.01)
	
	def forward(self, x):
		# residual = x
		
		# x_key = self.ResBlock(self.ResBlock(self.ResBlock(x)))
		
		x_value = self.GCBlock(x)
		# x_value = self.ResBlock(self.ResBlock(self.ResBlock(x_value)))
		# x_value = self.conv1x1(x_value)
		# x_value = torch.sigmoid(x_value)
		#
		# x_attention = torch.mul(x_key, x_value)
		#
		# out = residual + x_attention
		
		return x_value