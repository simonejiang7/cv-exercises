"""
develop a model based on the onnx file in model/model.onnx 

Note:
    - initialize the convolutions layer with uniform xavier
    - initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
    - initialize all biases with zeros
    - use batch norm wherever is relevant
    - use random seed 8
    - use default values for anything unspecified
"""

import numpy as np
import torch
import torch.nn as nn
import onnx
from onnx import numpy_helper
import torch
from onnx2torch import convert

torch.manual_seed(8)    # DO NOT MODIFY!
np.random.seed(8)   # DO NOT MODIFY!

model = onnx.load('model/model.onnx')
onnx.checker.check_model(model)
print("onnx model graph is => \n", onnx.helper.printable_graph(model.graph))
torch_model = convert('model/model.onnx')

# comment out the following lines to see the linear layer weights before initialization
# for name, param in torch_model.named_parameters():
#     if 'weight'in name and 'Gemm' in name:
#         print(f"{name}:\n{param.data}")

for name,param in torch_model.named_parameters():
    if 'weight'in name and 'Conv' in name:
        # initialize the convolutions layer with uniform xavier
        torch.nn.init.xavier_uniform_(param)
    elif 'weight'in name and 'Gemm' in name:
        # initialize the linear layer with a normal distribution (mean=0.0, std=1.0)
        torch.nn.init.normal_(param,mean=0.0,std=1.0)
    elif 'bias' in name:
        # initialize all biases with zeros
        torch.nn.init.zeros_(param)

# comment out the following lines to see the linear layer weights after initialization
# for name, param in torch_model.named_parameters():
#     if 'weight'in name and 'Gemm' in name:
#         print(f"{name}:\n{param.data}")

print("=> torch model before batch norm")
print(torch_model)

for name, module in torch_model.named_children():
    if 'Conv' in name:
        in_channels = module.out_channels
        bn = nn.BatchNorm2d(in_channels)
        # add batch norm layer after conv layer (however, there is option to add batch norm after activation layer, which is not implemented here)
        new_module = nn.Sequential(module, bn)
        setattr(torch_model, name, new_module)
    if 'Gemm' in name:
        in_channels = module.out_features
        bn = nn.BatchNorm1d(in_channels)
        # add batch norm layer after linear layer
        new_module = nn.Sequential(module, bn)
        setattr(torch_model, name, new_module)

print("=> torch model after batch norm")
print(torch_model)


      






    