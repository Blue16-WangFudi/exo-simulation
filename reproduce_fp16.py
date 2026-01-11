from tinygrad import Tensor, Device, dtypes
import os

print(f"Device: {Device.DEFAULT}")
try:
    # Force float16
    t = Tensor([1.0, 2.0], dtype=dtypes.float16).to("CUDA")
    t.realize()
    print("Success:", t.numpy())
except Exception as e:
    print("Failed:")
    print(e)
