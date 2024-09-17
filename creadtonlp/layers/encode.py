import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    def __init__(self, max_dim, max_seq):
        super(PositionalEncoding, self).__init__()
        self.max_dim, self.max_seq = max_dim, max_seq
        radian = self._get_angles(
            position=torch.arange(self.max_seq, dtype=torch.float32)[:, None],
            indices=torch.arange(self.max_dim, dtype=torch.float32)[None, :]
        )
        self.space = self._encode(radian)
        
    def forward(self, x):
        return x + self.space[:, :x.shape[1], :]
    
    def _get_angles(self, position, indices):
        numerator = 2 * (indices // 2)
        denominator = self.max_dim
        angles = 1 / torch.pow(10000, numerator / denominator)
        return position * angles
    
    def _encode(self, radian):
        sin = torch.sin(radian[:, 0::2])
        cos = torch.cos(radian[:, 1::2])
        result = torch.zeros_like(radian, dtype=torch.float32)
        result[:, 0::2] = sin
        result[:, 1::2] = cos
        result = result[None, ...]
        result.requires_grad = True
        return result

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    test = PositionalEncoding(128, 1024)
    
    plt.pcolormesh(test.space.cpu().detach().numpy()[0], cmap='RdBu')
    plt.xlabel('Dimension')
    plt.xlim((0, 128))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()