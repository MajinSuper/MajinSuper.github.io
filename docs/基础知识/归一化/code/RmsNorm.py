## file: RmsNorm.py
## author: majin
## date: 2025-09-05

import torch
import numpy as np

class RmsNorm(torch.nn.Module):
    def __init__(self, norm_shape, eps=1e-6,affine=True):
        super().__init__()
        self.eps = eps
        
        # 仿射变换
        self.affine = affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(norm_shape)) # [N]

    def forward(self, x):
        '''
            x: [B, L, N]
        '''
        var = torch.mean(x.pow(2),dim=-1,keepdim=True)  # [B, L, 1]
        x = x / torch.sqrt(var + self.eps) # [B, L, N]
        if self.affine:
            return x * self.weight
        else:
            return x


if __name__ == "__main__":
    
    x = torch.randn(1, 10, 256)
    rmsnorm = RmsNorm(256,eps=1e-5)
    with torch.no_grad():
        res = rmsnorm(x)
        print(res.shape)
        print(res)

    offical_rmsnorm = torch.nn.RMSNorm(256,eps=1e-5)
    with torch.no_grad():
        offical_rmsnorm.weight.data = torch.from_numpy(rmsnorm.weight.detach().numpy())
        offical_res = offical_rmsnorm(x)
        print(offical_res.shape)
        print(offical_res)
    
    allclose_res = torch.allclose(res, offical_res)
    print(allclose_res)
        