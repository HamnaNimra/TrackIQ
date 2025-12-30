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