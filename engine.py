class Value:
    def __init__(self, data, parents=(), operator=''):
        self.data = data
        self.parents = parents
        self.operator = operator
        self.grad = 0
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), 'add')
    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data - other.data, (self, other), 'sub')
    def square(self):
        return Value(self.data * self.data, (self, ), 'square')
    def relu(self):
        return Value(self.data if self.data > 0 else 0, (self, ), 'relu')
    def __repr__(self):
        return f'{self.data}'
    def __mul__(self, other):
        return Value(self.data * other.data, (self, other), 'mul')
    def backward(self):
        self.grad = 1
        visited = set()
        lst = list()
        self._get_prop_order(visited, lst)
        for node in reversed(lst):
            node._backprop()
    def _get_prop_order(self, visited, lst):
        if self in visited:
            return
        for node in self.parents:
            node._get_prop_order(visited, lst)
        lst.append(self)    
        visited.add(self)        
    def _backprop(self):
        if self.operator == 'add':
            for node in self.parents:
                node.grad += self.grad
        elif self.operator == 'mul':
            a = self.parents[0]
            b = self.parents[1]
            a.grad += self.grad * b.data
            b.grad += self.grad * a.data
        elif self.operator == 'square':
            self.parents[0].grad += self.grad * 2 * self.parents[0].data
        elif self.operator == 'relu':
            self.parents[0].grad += self.grad if self.data > 0 else 0
        elif self.operator == 'sub':
            a = self.parents[0]
            b = self.parents[1]
            a.grad += self.grad
            b.grad -= self.grad
