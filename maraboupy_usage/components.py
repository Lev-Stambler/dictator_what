import torch

class Slicer(torch.nn.Module):
    def __init__(self, inp_dim: int, start_at: int, output_dim: int):
        super().__init__()
        self.slicer = torch.nn.Linear(inp_dim, output_dim)
        stacked = []
        if start_at > 0:
            stacked.append(torch.zeros((output_dim, start_at)))
        stacked.append(torch.eye(output_dim))
        if start_at + output_dim < inp_dim:
            stacked.append(torch.zeros((output_dim, inp_dim - output_dim - start_at)))
        self.slicer.weight = torch.nn.Parameter(
            torch.hstack(stacked)
        )
    
    def forward(self, x):
        return self.slicer(x)

