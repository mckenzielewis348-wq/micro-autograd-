from engine import Value
from nn import MLP
from optim import SGD

# 1. Dataset & Model setup
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]

model = MLP(3, [4, 4, 1])

# 2. Instantiate Optimizer
optimizer = SGD(model.parameters(), lr=0.05)

# 3. Training Loop
for k in range(50):
    # Forward pass
    ypred = [model(x) for x in xs]
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Backward pass & update using optimizer
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if k % 10 == 0 or k == 49:
        print(f"Epoch {k:2d} | Loss: {loss.data:.4f}")
