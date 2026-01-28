"""PyTorch operations and a sample algorithm implementation.

Disclaimer: This code is for educational purposes only.
It demonstrates basic PyTorch operations and a simple algorithm.
Author: Hamna
Target: NVIDIA Edge AI / Automotive Performance Engineering

"""

import torch

# Check device
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {device}")

# Simple tensors
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

# Basic operation
c = a + b
print("Tensor c:", c)

# Simple linear model
model = torch.nn.Linear(3, 1)
x = torch.randn(5, 3)
y = model(x)

print("Model output:")
print(y)

# Backward pass test
loss = y.mean()
loss.backward()
print("Backward pass successful ✅")


def max_profit(prices):
    min_price = float("inf")
    best = 0
    for p in prices:
        if p < min_price:
            min_price = p
        else:
            best = max(best, p - min_price)
    return best
