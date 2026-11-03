from engine import Value
from random import uniform

class Neuron:
    def __init__(self, input_size):
        self.weights = []
        for i in range(input_size):
            self.weights.append(Value(uniform(-1, 1)))
        self.bias = Value(0)
    def parameters(self):
        res = []
        res.extend(self.weights)
        res.append(self.bias)
        return res
    def calc(self, x):
        if len(x) != len(self.weights):
            raise ValueError
        sum = Value(0)
        for i in range(len(x)):
            sum += self.weights[i] * x[i]
        return sum + self.bias

class Layer:
    def __init__(self, input_size, output_size):
        self.neurons = []
        for i in range(output_size):
            self.neurons.append(Neuron(input_size))
    def calc(self, inputs):
        res = []
        for neuron in self.neurons:
              res.append(neuron.calc(inputs))
        return res
    def parameters(self):
        res = []
        for neuron in self.neurons:
            res.extend(neuron.parameters())
        return res

class MLP:
    def __init__(self, layer_sizes):
        self.layers = []
        self.input_size = layer_sizes[0]
        self.output_size = layer_sizes[len(layer_sizes) - 1]
        for i in range(len(layer_sizes) - 1):
            layer = Layer(layer_sizes[i], layer_sizes[i+1])
            self.layers.append(layer)
    def parameters(self):
        res = []
        for layer in self.layers:
            res.extend(layer.parameters())
        return res
    def calc(self, inputs):
        for i in range(len(self.layers)):
            pre_activation = self.layers[i].calc(inputs)
            if i == len(self.layers) - 1:
                return pre_activation
            new_inputs = []
            for k in pre_activation:
                new_inputs.append(k.relu())
            inputs = new_inputs

