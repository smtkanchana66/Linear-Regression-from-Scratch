# Linear Regression from Scratch

A manual implementation of **linear regression with gradient descent** using only NumPy. no sklearn, no autograd, no shortcuts.

---

## What it does

- Predicts a continuous output from multiple input features
- Computes **Mean Squared Error (MSE)** as the loss function
- Updates weights and bias manually using **gradient descent**
- Tracks cost over epochs to verify convergence

---

## Math behind it

**Prediction:**
```
y_hat = w · x + b
```

**Cost (MSE):**
```
J = (1/n) * Σ (y_hat - y)²
```

**Gradients:**
```
∂J/∂w = (2/n) * Σ (y_hat - y) * x
∂J/∂b = (2/n) * Σ (y_hat - y)
```

**Weight update:**
```
w = w - lr * ∂J/∂w
b = b - lr * ∂J/∂b
```

---

## Project structure

```
├── Linear_regression.ipynb       # Main notebook with full implementation
└── README.md
```

---

## How to run

1. Clone the repo
```bash
git clone https://github.com/smtkanchana66/Linear-Regression-from-Scratch.git
cd Linear-Regression-from-Scratch
```

2. Install dependencies
```bash
pip install numpy
```

3. Open the notebook
```bash
jupyter notebook Linear_regression.ipynb
```

---

## Example

```python
X_train = np.array([
    [1, 5, 2, 8],
    [2, 3, 1, 6],
    [3, 7, 4, 2],
    [4, 1, 2, 9],
    [5, 4, 3, 1],
], dtype=float)

# y = 2*x1 + 3*x3
y_train = np.array([8, 7, 14, 14, 19], dtype=float)

w    = np.zeros(4)
bias = 0.0

for epoch in range(1000):
    predictions = pred_vec(X_train, w, bias)
    w, bias = gra(y_train, predictions, learning_rate=0.01, X_train=X_train, w=w, b=bias)
```

Expected weights after convergence: `w ≈ [2, 0, 3, 0]`, `bias ≈ 0`

---

## Key concepts practiced

- Forward pass (dot product prediction)
- Loss computation (MSE)
- Backpropagation by hand (chain rule)
- Gradient descent weight updates
- Importance of float dtype and feature scaling

---

## Requirements

- Python 3.x
- NumPy

---

## Author

Built from scratch as a learning exercise to understand the internals of linear regression without relying on ML libraries.
