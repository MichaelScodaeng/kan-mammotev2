from mamba_ssm import Mamba2
import torch
import time

# Create a random input tensor
x = torch.randn(1, 4, 256).to("cuda")
dim = 256

model = Mamba2(
    # This module uses roughly 3 * expand * d_model^2 parameters
    d_model=dim, # Model dimension d_model
    d_state=64,  # SSM state expansion factor, typically 64 or 128
    d_conv=4,    # Local convolution width
    expand=2,    # Block expansion factor
).to("cuda")

# warm up
y = model(x)

t1 = time.time()
y = model(x)
assert y.shape == x.shape
print(f"Time taken: {time.time() - t1:.3f} s")