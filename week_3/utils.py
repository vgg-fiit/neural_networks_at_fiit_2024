import numpy as np
from collections import OrderedDict


class Module:
    def __init__(self):
        self.modules = OrderedDict()
        self._parameters = OrderedDict()

    def add_module(self, module, name:str):
        if hasattr(self, name) and name not in self.modules:
            raise KeyError("attribute '{}' already exists".format(name))
        elif '.' in name:
            raise KeyError("module name can't contain \".\"")
        elif name == '':
            raise KeyError("module name can't be empty string \"\"")
        self.modules[name] = module

    def register_parameter(self, name, param):
        if '.' in name:
            raise KeyError("parameter name can't contain \".\"")
        elif name == '':
            raise KeyError("parameter name can't be empty string \"\"")
        elif hasattr(self, name) and name not in self._parameters:
            raise KeyError("attribute '{}' already exists".format(name))
        else:
            self._parameters[name] = param

    def parameters(self, recurse=True):
        for name, param in self._parameters.items():
            if param.requires_grad:
                yield name, param
        if recurse:
            for name, module in self._modules.items():
                for name, param in module.parameters(recurse):
                    if param.requires_grad:
                        yield name, param

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        modules = list(self._modules.keys())
        parameters = list(self._parameters.keys())
        keys = module_attrs + attrs + modules + parameters

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def __getattr__(self, name: str):
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        if '_parameters' in self.__dict__:
            parameters = self.__dict__['_parameters']
            if name in parameters:
                return parameters[name]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

    def __setattr__(self, name, value):
        if isinstance(value, Module):
            self.modules[name] = value
        elif isinstance(value, np.ndarray):
            self.register_parameter(name, value)
        else:
            object.__setattr__(self, name, value)

    def forward(self, *args, **kwargs) -> np.ndarray:
        pass

    def backward(self, *args, **kwargs) -> np.ndarray:
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


def gradient_check(network:Module, loss_function:Module, X:np.ndarray, Y:np.ndarray, epsilon=1e-7):
    # https: // datascience - enthusiast.com / DL / Improving_DeepNeural_Networks_Gradient_Checking.html
    # Set-up variables
    gradapprox = []
    grad_backward = []

    for name, layer in network.modules.items():
        # Compute gradapprox
        if not hasattr(layer, "W"):
            continue
        if not hasattr(layer, "dW"):
            continue
        shape = layer.W.shape
        # print(shape[0], ',', shape[1])
        for i in range(shape[0]):
            for j in range(shape[1]):
                # print('i',i,'j',j)
                # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]".
                # "_" is used because the function you have to outputs two parameters but we only care about the first one
                origin_W = np.copy(layer.W[i][j])

                layer.W[i][j] = origin_W + epsilon
                A_plus = network(X)
                J_plus = np.mean(loss_function(A_plus, Y))

                # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]".
                layer.W[i][j] = origin_W - epsilon
                A_minus = network(X)
                J_minus = np.mean(loss_function(A_minus, Y))

                # Compute gradapprox[i]
                gradapprox.append((J_plus - J_minus) / (2 * epsilon))
                # print(layer.name, layer.dW.shape)
                # grad = np.mean(layer.dW, axis=0, keepdims=True)
                # grad_backward.append(grad[0][i][j])
                grad_backward.append(layer.dW[i][j])
                layer.W[i][j] = origin_W

    # Compare gradapprox to backward propagation gradients by computing difference.
    gradapprox = np.reshape(gradapprox, (-1, 1))
    grad_backward = np.reshape(grad_backward, (-1, 1))

    numerator = np.linalg.norm(grad_backward - gradapprox)
    denominator = np.linalg.norm(grad_backward) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference > 2e-7 or not difference:
        print(
            "\033[91m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print(
            "\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")