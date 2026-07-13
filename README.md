# Neural Network from Scratch

A feedforward neural network implemented using only Python and NumPy.
No TensorFlow. No PyTorch. Just matrix math, backpropagation, and gradient descent.

## Files

| File            | Purpose                                                    |
| --------------- | ---------------------------------------------------------- |
| `neural_net.py` | The main implementation — a reusable `NeuralNetwork` class |

## Features

- **Configurable architecture** — pass any list of layer sizes (e.g. `[2, 8, 4, 1]`)
- **Multiple activations** — sigmoid, ReLU, tanh
- **Xavier initialization** — helps training converge faster than plain random weights
- **MSE loss** with vanilla gradient descent
- **Two demos** — XOR (the classic) and circle classification

## Quick start

```bash
# just run it
python neural_net.py
```

Output:

```
==================================================
  XOR  –  the classic nonlinear benchmark
==================================================

  epoch      1/10000   loss: 0.268292
  epoch   1000/10000   loss: 0.007248
  ...
  epoch  10000/10000   loss: 0.000152

Predictions:
  [0 0]  →  0.0124  ✓
  [0 1]  →  0.9867  ✓
  [1 0]  →  0.9869  ✓
  [1 1]  →  0.0144  ✓
```

## How it works

**Forward pass:** Input flows through the network layer by layer. Each layer does a weighted sum of its inputs, adds a bias, and passes the result through an activation function.

```
z = X · W + b
a = activation(z)
```

**Backward pass (backpropagation):** Starting from the output, we compute how much each weight contributed to the error using the chain rule. Then we nudge every weight in the direction that reduces the loss.

**Gradient descent:** The nudge size is controlled by the learning rate. Too big and training oscillates. Too small and it takes forever.

## Things I learned building this

1. You _need_ a hidden layer for XOR — a single-layer perceptron literally can't solve it
2. Weight initialization matters — all-zeros means all neurons learn the same thing
3. The learning rate is touchy — 2.0 diverged, 0.01 barely moved, 0.5–1.0 worked
4. Backprop is just the chain rule applied repeatedly — scarier in theory than in code
5. Shape bugs are the #1 time sink — print `.shape` after every operation

## Requirements

- Python 3.8+
- NumPy
