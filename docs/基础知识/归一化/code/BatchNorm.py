import torch


class MyBatchNorm1d(torch.nn.Module):
    def __init__(self, feature_num: int, eps=1e-5, momentum=0.1, affine=True):
        super().__init__()
        self.feature_num = feature_num
        self.eps = eps  # 防止除0

        # moving average
        self.momentum = momentum
        self.running_mean = torch.zeros(feature_num)  # [N]
        self.running_var = torch.ones(feature_num)  # [N]

        # 放射变换
        self.affine = affine
        if self.affine:
            self.gamma = torch.nn.Parameter(torch.ones(feature_num))  # [N]
            self.beta = torch.nn.Parameter(torch.zeros(feature_num))  # [N]

    def forward(self, x):
        """
        x: [B，N] or [B, N, L]
        """

        if x.dim() == 2:
            reduce_dims = [0]
            new_shape = [1, x.shape[1]]
        elif x.dim() == 3:
            reduce_dims = [0, 2]
            new_shape = [1, x.shape[1], 1]
        else:
            raise ValueError(f"Excepted 2D or 3D tensor , but get {x.dim()}")

        if self.training:
            mean_value = x.mean(dim=reduce_dims, keepdims=True)  # [1,N] 、[1,N,1]
            var_value = x.var(dim=reduce_dims, unbiased=False, keepdims=True)  # [1,N] 、[1,N,1]

            x_norm = (x - mean_value) / torch.sqrt(var_value + self.eps)  # [B,N]、 [B,N,L]

            # moving mean and var
            self.running_mean = self.running_mean * (1 - self.momentum) + mean_value.view(-1) * self.momentum  # [N]             
            self.running_var =  self.running_var * (1 - self.momentum)  + var_value.view(-1) * self.momentum # [N]
        else:
            x_norm = ( x - self.running_mean.reshape(new_shape) ) / self.running_var.reshape(new_shape)

        if self.affine:
            return x_norm * self.gamma.reshape(new_shape) + self.beta.reshape(new_shape)
        else:
            return x_norm


def eval_1d(output_prefix, input_tensor):

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

    print(
        f"{output_prefix}, [train] the result is close: {torch.allclose(my_res,gf_res,1e-4)}"
    )

    ######################################################
    # for eval step
    ######################################################
    my_bn_layer.eval()
    my_res = my_bn_layer(input_tensor)

    gf_bn_layer.eval()
    gf_res = gf_bn_layer(input_tensor)

    print(f"{output_prefix}, [eval] the result is close: {torch.allclose(my_res,gf_res,1e-4)}")

def eval_dim_3to2(input_tensor):
    B,N,L = input_tensor.shape[:]

    my_bn_layer = MyBatchNorm1d(input_tensor.shape[1]) # [B,N,L]
    my_bn_layer.train()
    gf_bn_layer = torch.nn.BatchNorm1d(input_tensor.shape[1])

    my_3d_res = my_bn_layer(input_tensor) # [B,N,L]

    input_tensor_3d_to_2d = input_tensor.clone().permute(0,2,1).contiguous().view(-1, N) # [B,N,L] -> [B*L,N]
    my_3to2_res = my_bn_layer(input_tensor_3d_to_2d) # [B*L,N]

    my_3to2_res = my_3to2_res.view(B,L,-1).permute(0,2,1) # [B*L,N] , [B,L,N], [B,N,L]
    
    print(my_3d_res)
    print(my_3to2_res)
    print(f"{torch.allclose(my_3d_res,my_3to2_res,1e-4)}")



if __name__ == "__main__":
    # 比较2d结果 自己实现的 vs 官方实现的
    input_tensor = torch.rand([2, 4])
    print(f"input: {input_tensor}")
    eval_1d("2d", input_tensor)

    # 比较3d结果 自己实现的 vs 官方实现的
    input_tensor_3D = torch.rand([2, 4, 3])
    print(f"input: {input_tensor_3D}")
    eval_1d("3d", input_tensor_3D)
    print(f"input: {input_tensor_3D}")

    # 比较3d结果 3d上的norm等价与转成2d上的norm
    eval_dim_3to2(input_tensor_3D)