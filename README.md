# micro-autograd-

A lightweight scalar-based autograd engine and neural network built from scratch in Python.

## Overview

`micro-autograd-` implements a dynamic computation graph (similar to PyTorch) that tracks mathematical operations on scalar values and automatically calculates gradients using backpropagation.

## Architecture

* **`engine.py`**: Defines the `Value` node class, which tracks values, builds operational execution trees, and triggers backpropagation via topological sort.
* **`nn.py`**: Builds neural network abstractions (`Neuron`, `Layer`) using `Value` objects to simulate weight updates and forward passes.
* **`train.py`**: Demonstrates forward passes, Mean Squared Error (MSE) loss computation, autograd backpropagation, and Stochastic Gradient Descent (SGD) optimization.

## How to Run

Run the training script in your terminal:

```bash
python3 train.py
