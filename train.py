from model import MLP
from engine import Value

class Train:
    def __init__(self, learning_rate=0.01, round=10):
        self.learning_rate = learning_rate
        self.round = round
    def train(self, model, samples):
        if not samples:
            return
        for sample in samples:
            if model.input_size != len(sample[0]):
                raise ValueError
            if model.output_size != len(sample[1]):
                raise ValueError
        for i in range(self.round):
            loss = Value(0)
            for param in model.parameters():
                param.grad = 0
            for sample in samples:
                input_values = []
                expect_values = []
                for v in sample[0]:
                    input_values.append(Value(v))
                for v in sample[1]:
                    expect_values.append(Value(v))
                predict_values = model.calc(input_values)
                for i in range(len(expect_values)):
                    loss += (expect_values[i] - predict_values[i]).square()
            loss.backward()
            for param in model.parameters():
                param.data = param.data - self.learning_rate * param.grad
