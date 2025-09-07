import torch 
class MyLayerNorm(torch.nn.Module):
    def __init__(self,batch_size,eps=1e-5,affine=True) -> None:
        super().__init__()
        self.eps = eps

        # 放射变换
        self.affine =  affine
        if self.affine:
            self.weight = torch.nn.Parameter(torch.ones(feature_num)) #[N]
            self.bias = torch.nn.Parameter(torch.zeros(feature_num)) #[N]

    def forward(self,x):
        '''
            x : [B, L, N] or [B, N]
        '''
        mean_value = x.mean(dim=-1,keepdim=True) # [B, 1] [B, 1, L]
        var_value = x.var(dim=-1,unbiased=False,keepdim=True) # [B, 1] [B, 1, L]

        x_norm = (x - mean_value) / torch.sqrt(var_value+self.eps) # [B, N] [B, N, L]

        if self.affine:
            return x_norm * self.weight + self.bias
        else:
            return x_norm


if __name__ == "__main__":
    batch_size = 4
    feature_num = 2

    gf_ln_layer = torch.nn.LayerNorm(feature_num,eps=1e-5)
    my_ln_layer = MyLayerNorm(feature_num)
    
    gf_ln_layer.weight.data = my_ln_layer.weight.detach().clone()
    gf_ln_layer.bias.data = my_ln_layer.bias.detach().clone()

    # 2d
    input_tensor = torch.rand([batch_size,feature_num])
    gf_res_2d = gf_ln_layer(input_tensor)
    my_res_2d = my_ln_layer(input_tensor)
    print(f" 2d layer norm result is :{torch.allclose(gf_res_2d,my_res_2d,1e-4)}")
    
    # 3d
    seq_len = 5
    input_tensor_3d = torch.rand([batch_size,seq_len,feature_num])
    my_res_3d= my_ln_layer(input_tensor_3d)
    gf_res_3d = gf_ln_layer(input_tensor_3d)
    
    print(my_res_3d)
    print(gf_res_3d)
    print(f" 3d layer norm result is :{torch.allclose(my_res_3d,gf_res_3d,atol=1e-6)}")