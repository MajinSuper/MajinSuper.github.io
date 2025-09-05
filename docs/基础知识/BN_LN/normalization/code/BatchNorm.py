import torch

class MyBatchNorm1d(torch.nn.Module):
    def __init__(self, feature_num, eps=1e-5, momentum=0.1,affine= True):
        super().__init__()
        self.feature_num = feature_num
        self.eps = eps
        self.momentum = momentum
        self.affine =affine
        self.running_mean = torch.zeros(feature_num)
        self.running_var = torch.ones(feature_num)
        
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.zeros(feature_num))
            self.beta = torch.nn.Parameter(torch.ones(feature_num))
    
    def forward(self, x):
        '''
            x: [B, N, L] or [B，N]
        '''
        
        if x.dim()==2:
            reduce_dims=[0]
            affine_output_shape = [1,x.shape[1]]

        elif x.dim() == 3:
            reduce_dims=[0,2]
            affine_output_shape = [1,x.shape[1],1]

        else:
            raise ValueError(f"Excepted 2D or 3D tensor , but get {x.dim()}")

        if self.training:            
            mean_value = x.mean(dim=reduce_dims,keepdims=True) # [1,N] 、[1,N,1]
            var_value = x.var(dim=reduce_dims,unbiased=False,keepdims=True)  # [1,N] 、[1,N,1]           
            
            x_norm = (x - mean_value )/ (torch.sqrt(var_value + self.eps)) # [B,N,L]
            
            # 滑动平均
            self.running_mean = self.running_mean * (1-self.momentum) + mean_value.view(-1) *self.momentum # [N]
            self.running_var = self.running_var * (1-self.momentum) + var_value.view(-1) *self.momentum #[N]
        else:

            x_norm =  (x-self.running_mean.reshape(affine_output_shape)) / self.running_var.reshape(affine_output_shape)

        if self.affine:
            return x_norm * self.gamma.reshape(affine_output_shape) + self.beta.reshape(affine_output_shape)
        else:
            return x_norm

def eval_1d(output_prefix , input_tensor):

    my_bn_layer = MyBatchNorm1d(input_tensor.shape[1])
    gf_bn_layer = torch.nn.BatchNorm1d(input_tensor.shape[1])
    
    ######################################################
    # for train step
    ######################################################
    my_bn_layer.train()
    my_res = my_bn_layer(input_tensor)

    gf_bn_layer.train()
    gf_bn_layer.weight = my_bn_layer.gamma
    gf_bn_layer.bias = my_bn_layer.beta
    gf_res = gf_bn_layer(input_tensor)

    print(f"{output_prefix}, [train] the result is close: {torch.allclose(my_res,gf_res,1e-4)}")

    ######################################################
    # for eval step
    ######################################################
    my_bn_layer.eval()
    my_res = my_bn_layer(input_tensor)

    gf_bn_layer.eval()
    gf_res = gf_bn_layer(input_tensor)

    print(f"{output_prefix}, [eval] the result is close: {torch.allclose(my_res,gf_res,1e-4)}")



if __name__ == "__main__":
    # input_tensor = torch.rand([2,4])
    # print(f"input: {input_tensor}")
    # eval_1d("2d",input_tensor)

    input_tensor_3D = torch.rand([2,4,3])
    print(f"input: {input_tensor_3D}")
    eval_1d("3d",input_tensor_3D)