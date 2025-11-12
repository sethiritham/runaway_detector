import torch
print(f"Is CUDA available? {torch.cuda.is_available()}")
print(f"CUDA version PyTorch was built with: {torch.version.cuda}")