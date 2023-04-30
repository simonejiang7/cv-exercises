"""
Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution. 

The common way of using grouped 2D convolution layer in Pytorch is to use 
torch.nn.Conv2d(groups=n), where n is the number of groups.

However, it is possible to use a stack of n torch.nn.Conv2d(groups=1) to replicate the same
result. The weights must be copied and be split between the convs in the stack.

You can use:
    - use default values for anything unspecified  
    - all available functions in NumPy and Pytorch
    - the custom layer must be able to take all parameters of the original nn.Conv2d 
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

# random input (batch, channels, height, width)
x = torch.randn(2, 64, 100, 100)

# original 2d convolution
grouped_layer = nn.Conv2d(64, 128, 3, stride=1, padding=1, groups=16, bias=True)

# weights and bias
w_torch = grouped_layer.weight
b_torch = grouped_layer.bias

y = grouped_layer(x)

class GroupedConv2D(nn.Module):

    def __init__(self, input_channels: int = 64, output_channels: int = 128, kernel_size: int = 3, groups: int = 16):
        super(GroupedConv2D, self).__init__()
        self.conv = nn.Conv2d(input_channels, output_channels, kernel_size, stride=1, padding=1, groups=groups, bias=True)
        
    def forward(self, x):
        x = self.conv(x)
        return x

# now write your custom layer
class CustomGroupedConv2D(nn.Module):
    '''
    Write a code using pytorch to replicate a grouped 2D convolution layer based on the original 2D convolution.
    Without grouping: 128 Filters, each with 64 kernels of size 3x3
    With grouping: 128 Filters, grouped into 16 groups, each with 4 kernels of size 3x3
    '''

    def __init__(self, input_channels: int = 64, output_channels: int = 128, kernel_size: int = 3, groups: int = 16):

        super(CustomGroupedConv2D, self).__init__()
        self.convs = nn.ModuleList([nn.Conv2d(input_channels//groups, output_channels//groups, kernel_size, stride=1, padding=1, groups=1, bias=True) for _ in range(groups)])
        self.groups = groups
        self.input_channels = input_channels
        self.output_channels = output_channels

        # copy and split weights and bias into 16 groups
        with torch.no_grad():
            for i, conv in enumerate(self.convs):
                # w_torch.shape = torch.Size([128, 4, 3, 3])
                conv.weight.data = w_torch[i * output_channels//groups : (i+1) * output_channels//groups].data
                # b_torch.shape = torch.Size([128])
                conv.bias.data = b_torch[i * output_channels//groups : (i+1) * output_channels//groups].data
            
    def forward(self, x):

        # split input into groups
        input_groups = torch.split(x, x.size(1) // self.groups, dim=1)
        # make sure the input is split into groups of correct shape
        assert input_groups[0].shape == torch.Size([2, int(self.input_channels/self.groups), 100, 100])
        # apply convolutions to each group
        output_groups = [conv(group) for conv, group in zip(self.convs, input_groups)]
        # concatenate the output groups of shape [16,100,100] to [128,100,100]
        output = torch.cat(output_groups, dim=1)
        assert output.shape == torch.Size([2, self.output_channels, 100, 100])
        
        return output
    

# the output of CustomGroupedConv2D(x) must be equal to grouped_layer(x)
custom_grouped_layer = CustomGroupedConv2D()
assert torch.allclose(grouped_layer(x), custom_grouped_layer(x),rtol=1e-3, atol=1e-6)
print("Test Passed! The output of CustomGroupedConv2D(x) is equal to grouped_layer(x)")








        
