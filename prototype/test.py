import sys
import torch
import transformers

print("Python:", sys.version)
print("Executable:", sys.executable)
print("PyTorch:", torch.__version__)
print("Torch:", torch.__file__)
print("Transformers:", transformers.__version__)
print("Transformers sees PyTorch:", transformers.is_torch_available())
print("MPS:", torch.backends.mps.is_available())