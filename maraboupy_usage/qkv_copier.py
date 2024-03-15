
import torch
from components import Slicer

class QKVCopier(torch.nn.Module):
    def __init__(self, seq_len=2, num_heads=8, hidden_size=64) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        i = 0
        self.slicer_0  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)
        i = 1
        self.slicer_1  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)
        i = 2
        self.slicer_2  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)
        i = 3
        self.slicer_3  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)
        i = 4
        self.slicer_4  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)
        i = 5
        self.slicer_5  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)
        i = 6
        self.slicer_6  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)
        i = 7
        self.slicer_7  =Slicer(num_heads * 3 * hidden_size, i * 3 * hidden_size, 3 * hidden_size)
