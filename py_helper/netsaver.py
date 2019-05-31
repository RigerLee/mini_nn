import numpy as np
import re
import os
import shutil
import torch
import torch.nn as nn
from leNet import Net

class NetSaver(object):
    def __init__(self, net, legal_input):
        self.net_ = net
        self.legal_input_ = legal_input
        self.counter_ = 0
        # mkdir and store weights
        self.file_path_ = './model/'
        if os.path.exists(self.file_path_):
            shutil.rmtree(self.file_path_)
        os.makedirs(self.file_path_)

    # Get forward layers and return
    def get_forward_pass(self):
        var = self.net_(self.legal_input_)
        seen = set()
        forward = []
        # recursively find next forward layer
        def add_nodes(var):
            if var not in seen:
                if torch.is_tensor(var):
                    pass
                elif hasattr(var, 'variable'):
                    pass
                else:
                    forward.append(str(type(var).__name__))
                seen.add(var)
                if hasattr(var, 'next_functions'):
                    for u in var.next_functions:
                        if u[0] is not None:
                            add_nodes(u[0])
                if hasattr(var, 'saved_tensors'):
                    for t in var.saved_tensors:
                        add_nodes(t)
        add_nodes(var.grad_fn)
        forward.reverse()
        write_layers = []
        # check layer type
        for layer in forward:
            if re.search('Convolution',layer):
                write_layers.append('Conv2d')
            elif re.search('Relu',layer):
                write_layers.append('ReLU')
            elif re.search('MaxPool2D',layer):
                write_layers.append('MaxPool2d')
            elif re.search('Addmm',layer):
                write_layers.append('Linear')
        return write_layers

    # get layers and it's parameters and save weights and bias
    def deal_with_layers(self):
        net = self.net_
        layer_list = []
        for layer in net.modules():
            if isinstance(layer, nn.Linear):
                result = re.search('(?:Linear\()(?:\D+)(\d+)(?:\D+)(\d+)(?:\D+)', str(layer))
                if result is None:
                    exit(-1)
                result = list(result.groups())
                weight_name = self.file_path_ + str(self.counter_) + '.npy'
                bias_name = self.file_path_ + str(self.counter_ + 1)  + '.npy'
                self.counter_ += 2
                # save weights
                np.save(weight_name, layer.weight.detach().numpy().reshape((1,-1)))
                bias = layer.bias.detach().numpy().reshape((1,-1))
                np.save(bias_name, bias)
                # record data type and file name
                result.append(str(bias.dtype))
                result.append(weight_name)
                result.append(bias_name)
                layer_list.append(['Linear'] + result)
            elif isinstance(layer, nn.Conv2d):
                result = re.search('(?:Conv2d\()(\d+)(?:\D+)(\d+)(?:\D+)(\d+)(?:\D+)(?:\d+)(?:\D+)(\d+)(?:\D+)(?:\d+)(?:\D+)(\d+)(?:\D+)(?:\d+)(?:\D+)', str(layer))
                if result is None:
                    result = re.search('(?:Conv2d\()(\d+)(?:\D+)(\d+)(?:\D+)(\d+)(?:\D+)(?:\d+)(?:\D+)(\d+)(?:\D+)(?:\d+)(?:\D+)', str(layer))
                if result is None:
                    exit(-1)
                result = list(result.groups())
                weight_name = self.file_path_ + str(self.counter_) + '.npy'
                bias_name = self.file_path_ + str(self.counter_ + 1)  + '.npy'
                self.counter_ += 2
                # save weights
                np.save(weight_name, layer.weight.detach().numpy().reshape((1,-1)))
                bias = layer.bias.detach().numpy().reshape((1,-1))
                np.save(bias_name, bias)
                # record data type and file name
                result.append(str(bias.dtype))
                result.append(weight_name)
                result.append(bias_name)
                layer_list.append(['Conv2d'] + result)
            elif isinstance(layer, nn.MaxPool2d):
                result = re.search('(?:MaxPool2d\()(?:\D+)(\d+)(?:\D+)(\d+)(?:\D+)(\d+)(?:.*)', str(layer))
                if result is None:
                    exit(-1)
                result = list(result.groups())
                layer_list.append(['MaxPool2d'] + result)
        return layer_list

    def run(self):
        layers = self.get_forward_pass()
        layer_params = self.deal_with_layers()
        #print('\n'.join([' '.join(i) for i in layer_params]))
        # substitute layer by layer_param
        for i in range(len(layers)):
            if len(layer_params) > 0 and layers[i] == layer_params[0][0]:
                layers[i] = ' '.join(layer_params[0])
                layer_params.pop(0)
        layers = '\n'.join(layers)
        with open(self.file_path_ + 'layer.txt',mode='w') as f:
            f.write(layers + '\n')
        return layers
        
x = np.linspace(-1.0, 1.0, 784, dtype='float32').reshape((1, 1, 28, 28))
x = torch.from_numpy(x)
model = torch.load('./temp.pth')
temp = NetSaver(model, x)
layers = temp.run()
# print the structure of neural network
#print(layers)

exit()







