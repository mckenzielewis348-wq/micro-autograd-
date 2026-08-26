class SGD:
    def __init__(self, params, lr=0.01):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        """Reset gradients for all parameters to 0.0 before a backward pass."""
        for p in self.params:
            p.grad = 0.0

    def step(self):
        """Update parameter values based on computed gradients."""
        for p in self.params:
            p.data -= self.lr * p.grad
