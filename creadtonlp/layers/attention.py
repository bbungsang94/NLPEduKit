import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, max_dim, max_seq, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert max_dim % n_heads == 0
        
        self.max_dim = max_dim
        self.max_seq = max_seq
        self.n_heads = n_heads
        self.sub_dim = self.max_dim // self.n_heads
        
        self.weight_dict = nn.ModuleDict({
            'query': nn.Linear(in_features=self.max_seq, out_features=self.max_dim),
            'key': nn.Linear(in_features=self.max_seq, out_features=self.max_dim),
            'value': nn.Linear(in_features=self.max_seq, out_features=self.max_dim),
            'output': nn.Linear(in_features=self.max_seq, out_features=self.max_dim)
        })
    
    def forward(self, x):
        pass
    
    def _split_heads(self, qkv_tuple, batch_size):
        pass
    
    def _extract_vectors(self, qkv_tuple):
        pass
    