import numpy as np

def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def _sigmoid_prime(a):
    return a * (1.0 - a)

def _relu(z):
    return np.maximum(0, z)

def _relu_prime(a):
    return (a > 0).astype(float)

def _tanh(z):
    return np.tanh(z)

def _tanh_prime(a):
    return 1.0 - a ** 2


_ACTIVATIONS = {
    "sigmoid": (_sigmoid, _sigmoid_prime),
    "relu":    (_relu,    _relu_prime),
    "tanh":    (_tanh,    _tanh_prime),
}


class NeuralNetwork:

    def __init__(self, layer_sizes, activation="sigmoid", seed=None):
        if activation not in _ACTIVATIONS:
            raise ValueError(f"Unknown activation '{activation}'. "
                             f"Pick from {list(_ACTIVATIONS)}")

        self.layer_sizes = layer_sizes
        self.act_fn, self.act_deriv = _ACTIVATIONS[activation]
        self.rng = np.random.default_rng(seed)

        self.weights = []
        self.biases = []
        for i in range(len(layer_sizes) - 1):
            fan_in = layer_sizes[i]
            fan_out = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (fan_in + fan_out))
            w = self.rng.uniform(-limit, limit, size=(fan_in, fan_out))
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

        self.loss_history = []


    def _forward(self, X):

        activations = [X]
        current = X

        for w, b in zip(self.weights, self.biases):
            z = current.dot(w) + b
            current = self.act_fn(z)
            activations.append(current)

        return activations

    def _backward(self, activations, y, lr):
        
        m = y.shape[0]  # number of training examples
        delta = (activations[-1] - y) * self.act_deriv(activations[-1])

        # walk backwards through the layers
        for i in reversed(range(len(self.weights))):
            grad_w = activations[i].T.dot(delta) / m
            grad_b = np.sum(delta, axis=0, keepdims=True) / m

            # propagate delta to previous layer (only if not at the input)
            if i > 0:
                delta = delta.dot(self.weights[i].T) * self.act_deriv(activations[i])

            self.weights[i] -= lr * grad_w
            self.biases[i] -= lr * grad_b

    def train(self, X, y, epochs=5000, lr=0.5, verbose=True):

        self.loss_history = []
        print_every = max(1, epochs // 10)

        for epoch in range(1, epochs + 1):
            activations = self._forward(X)
            loss = np.mean((activations[-1] - y) ** 2)
            self.loss_history.append(loss)
            self._backward(activations, y, lr)

            if verbose and (epoch % print_every == 0 or epoch == 1):
                print(f"  epoch {epoch:>6d}/{epochs}   loss: {loss:.6f}")


    def predict(self, X):
        return self._forward(X)[-1]


def demo_xor():
    print("=" * 50)
    print("  XOR  –  the classic nonlinear benchmark")
    print("=" * 50)
    print()

    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork([2, 4, 1], activation="sigmoid", seed=42)
    nn.train(X, y, epochs=10_000, lr=1.0)

    print("\nPredictions:")
    preds = nn.predict(X)
    for i in range(len(X)):
        raw = preds[i][0]
        label = "✓" if round(raw) == y[i][0] else "✗"
        print(f"  {X[i]}  →  {raw:.4f}  {label}")

    print(f"\nFinal loss: {nn.loss_history[-1]:.6f}")
    return nn


def demo_circle():

    print("\n")
    print("=" * 50)
    print("  Circle classification  –  nonlinear boundary")
    print("=" * 50)
    print()

    rng = np.random.default_rng(7)
    n = 200
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] ** 2 + X[:, 1] ** 2) < 0.5).astype(float).reshape(-1, 1)

    nn = NeuralNetwork([2, 8, 4, 1], activation="sigmoid", seed=7)
    nn.train(X, y, epochs=5000, lr=2.0)

    preds = nn.predict(X)
    accuracy = np.mean((preds > 0.5).astype(float) == y)
    print(f"\nAccuracy: {accuracy * 100:.1f}%")
    return nn


if __name__ == "__main__":
    demo_xor()
    demo_circle()
