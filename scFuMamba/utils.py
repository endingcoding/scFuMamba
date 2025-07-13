import torch                    
import torch.nn as nn             
import torch.nn.functional as F     
import numpy as np                  
from einops import repeat

# preprocessing
def random_indexes(size : int):
    forward_indexes = np.arange(size)
    np.random.shuffle(forward_indexes)
    backward_indexes = np.argsort(forward_indexes)
    return forward_indexes, backward_indexes

def take_indexes(sequences, indexes):
    return torch.gather(sequences, 0, repeat(indexes, 't b -> t b c', c=sequences.shape[-1]))

class PatchShuffle(torch.nn.Module):
    def __init__(self, ratio):
        super().__init__()
        self.ratio = ratio

    def forward(self, patches : torch.Tensor):
        T, B, C = patches.shape
        remain_T = int(T * (1 - self.ratio))
        
        indexes = [random_indexes(T) for _ in range(B)]
        forward_indexes = torch.as_tensor(np.stack([i[0] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)
        backward_indexes = torch.as_tensor(np.stack([i[1] for i in indexes], axis=-1), dtype=torch.long).to(patches.device)

        patches = take_indexes(patches, forward_indexes)
        patches = patches[:remain_T]

        return patches, forward_indexes, backward_indexes
    
class ECALayer(nn.Module):
    def __init__(self, channel, k_size=3):
        super(ECALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.transpose(-1, -2)).transpose(-1, -2)
        y = self.sigmoid(y)
        return x * y

class LDC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LDC, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.descriptive_weight = nn.Parameter(torch.randn(out_channels, 1))
    
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        descriptive_weight = self.descriptive_weight.expand_as(x)
        x = x * descriptive_weight
        x = x.permute(0, 2, 1)
        return x

class GlobalAveragePooling1d(nn.Module):
    def __init__(self):
        super(GlobalAveragePooling1d, self).__init__()
    
    def forward(self, x):
        return torch.mean(x, dim=2, keepdim=True)
    
class MaxPoolAdjust(nn.Module):
    def __init__(self):
        super(MaxPoolAdjust, self).__init__()

    def forward(self, rna_tensor, protein_tensor):
        batch_size, protein_seq_len, emb_dim = protein_tensor.shape
        _, rna_seq_len, _ = rna_tensor.shape
        kernel_size = rna_seq_len // protein_seq_len
        max_pool = nn.MaxPool1d(kernel_size, stride=kernel_size, padding=0)
        rna_tensor_pooled = max_pool(rna_tensor.transpose(1, 2)).transpose(1, 2)
        return rna_tensor_pooled
    
class MinPooling1d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MinPooling1d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride if stride is not None else kernel_size
        self.padding = padding

    def forward(self, x):
        device = x.device

        x_padded = F.pad(x, (self.padding, self.padding), mode='constant', value=float('inf'))
        
        kernel = self.create_min_kernel(x.size(1)).to(device) 
        output = F.conv1d(x_padded, kernel, stride=self.stride, padding=0)
        
        return output

    def create_min_kernel(self, num_channels):
        kernel = torch.ones(num_channels, 1, self.kernel_size) * -1
        return kernel

    def create_min_kernel(self, num_channels):
        kernel = torch.ones(num_channels, 1, self.kernel_size) * -1
        return kernel