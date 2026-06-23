import torch
import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device("cuda")       # NVIDIA GPU
elif torch.backends.mps.is_available():
    device = torch.device("mps")        # Apple Silicon GPU
else:
    device = torch.device("cpu")

print(f"Using device: {device}")

x = torch.randn(3, 3, device=device)
print(x)

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 2)
    def forward(self, x):
        return self.fc(x)

# Instantiate and push to GPU
model = SimpleNet().to(device)