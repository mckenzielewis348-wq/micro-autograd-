from nn import Layer
from engine import Value

model = Layer(2, 1)

inputs = [[2.0, 3.0], [-1.0, -2.0], [1.5, 0.5]]
targets = [1.0, 0.0, 1.0]

learning_rate = 0.05

print("Starting Training Loop...")

for epoch in range(15):
    preds = [model(x) for x in inputs]
    loss = sum((p - t)**2 for p, t in zip(preds, targets))

    for p in model.parameters():
        p.grad = 0.0

    loss.backward()

    for p in model.parameters():
        p.data -= learning_rate * p.grad

    print(f"Epoch {epoch + 1:2d} | Loss: {loss.data:.4f}")

print("Training Complete!")
