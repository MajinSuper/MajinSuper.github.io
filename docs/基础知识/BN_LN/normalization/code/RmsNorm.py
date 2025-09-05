## file: RmsNorm.py
## author: majin
## date: 2025-09-05

import torch
import numpy as np

class RmsNorm(torch.nn.Module):
    def __init__(self, norm_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = torch.nn.Parameter(torch.ones(norm_shape))
        # self.bias = torch.nn.Parameter(torch.zeros(hidden_size,1))
    
    def forward(self, x):
        # x: [batch_size, seq_len, hidden_size]
        var = torch.mean(x.pow(2),dim=-1,keepdim=True)  # [batch_size, seq_len, 1]
        x = x / torch.sqrt(var + self.eps) # [batch_size, seq_len, hidden_size]

        return x * self.weight.unsqueeze(0).to(x.dtype)


if __name__ == "__main__":
    
    x = torch.randn(1, 10, 256)
    rmsnorm = RmsNorm([10,256],eps=1e-5)
    with torch.no_grad():
        res = rmsnorm(x)
        print(res.shape)
        print(res)

    offical_rmsnorm = torch.nn.RMSNorm([10,256],eps=1e-5)
    with torch.no_grad():
        offical_rmsnorm.weight.data = torch.from_numpy(rmsnorm.weight.detach().numpy())
        offical_res = offical_rmsnorm(x)
        print(offical_res.shape)
        print(offical_res)
    
    allclose_res = torch.allclose(res, offical_res)
    print(allclose_res)
        