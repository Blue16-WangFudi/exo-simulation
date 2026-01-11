from tinygrad import Tensor, Device
import os

print(f"Device: {Device.DEFAULT}")
try:
    t = Tensor([1.0, 2.0]).to("CUDA")
    t.realize()
    print("Success:", t.numpy())
except Exception as e:
    print("Failed:")
    print(e)
