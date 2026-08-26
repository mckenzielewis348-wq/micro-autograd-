import random
from engine import Value
from nn import MLP

# 1. Dataset (4 inputs, 4 target outputs)
xs = [
    [2.0, 3.0, -1.0],
    [3.0, -1.0, 0.5],
    [0.5, 1.0, 1.0],
    [1.0, 1.0, -1.0],
]
ys = [1.0, -1.0, -1.0, 1.0]  # Desired targets

# 2. Instantiate Model (3 inputs -> two 4-neuron hidden layers -> 1 output)
model = MLP(3, [4, 4, 1])

# 3. Optimization Parameters
learning_rate = 0.05
epochs = 50

print(f"Starting training for {epochs} epochs...\n")

# 4. Training Loop
for k in range(epochs):

    # Forward pass: evaluate predictions
    ypred = [model(x) for x in xs]

    # Calculate Mean Squared Error (MSE) loss
    loss = sum((yout - ygt) ** 2 for ygt, yout in zip(ys, ypred))

    # Zero out gradients before backward pass
    for p in model.parameters():
        p.grad = 0.0

    # Backward pass: compute gradients via backprop
    loss.backward()

    # Update weights/biases (Gradient Descent)
    for p in model.parameters():
        p.data -= learning_rate * p.grad

    # Print loss progression
    if k % 10 == 0 or k == epochs - 1:
        print(f"Epoch {k:2d} | Loss: {loss.data:.4f}")

print("\nFinal Predictions vs Targets:")
for x, y_target, y_pred in zip(xs, ys, ypred):
    print(f"Input: {x} -> Target: {y_target: .1f} | Prediction: {y_pred.data: .4f}")
