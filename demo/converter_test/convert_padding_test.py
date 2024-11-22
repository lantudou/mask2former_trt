
from torch2trt.torch2trt import *
import torch
import torch.nn as nn
import torch.nn.functional as F

class Padtest(nn.Module):
    def __init__(self):
        super(Padtest, self).__init__()

    def forward(self, x):
        pad_l = 4
        pad_r = 10
        pad_t = 3
        pad_b = 5

        return   F.pad(x, (3, 3, pad_l, pad_r, pad_t, pad_b))

@tensorrt_converter('torch.nn.functional.pad')
def convert_pad(ctx):
    input_tensor = get_arg(ctx, 'x', pos=0, default=None)
    padding = get_arg(ctx, 'pad', pos=1, default=None)
    output = ctx.method_return

    # Get input shape
    N, H, W, C = input_tensor.shape

    # Extract padding values, skipping first two values which specify pad value=0
    pad_s, pad_e, pad_l, pad_r, pad_t, pad_b = padding
    assert pad_s >= 0 and pad_e >= 0 and pad_l >= 0  and  pad_r >= 0 and pad_t >= 0 and pad_b >= 0
    # Convert input tensor to TRT tensor
    input_tensor_trt = add_missing_trt_tensors(ctx.network, [input_tensor])[0]
    current_tensor = input_tensor_trt

    # Track current height manually
    current_H = H

    # Handle height padding (pad_t, pad_b)
    if pad_t > 0:
        data_top = np.zeros((N, pad_t, W, C), dtype=np.float32)
        pad_top = ctx.network.add_constant(tuple(data_top.shape), data_top)
        concat = ctx.network.add_concatenation([pad_top.get_output(0), current_tensor])
        concat.axis = 1  # Height axis
        current_tensor = concat.get_output(0)
        current_H += pad_t  # Update tracked height

    if pad_b > 0:
        data_bottom = np.zeros((N, pad_b, W, C), dtype=np.float32)
        pad_bottom = ctx.network.add_constant(tuple(data_bottom.shape), data_bottom)
        concat = ctx.network.add_concatenation([current_tensor, pad_bottom.get_output(0)])
        concat.axis = 1  # Height axis
        current_tensor = concat.get_output(0)
        current_H += pad_b  # Update tracked height

    current_W = W
    # Use calculated height for width padding
    if pad_l > 0:
        zero_data_left = np.zeros((N, current_H, pad_l, C), dtype=np.float32)
        pad_left = ctx.network.add_constant(tuple(zero_data_left.shape), zero_data_left)
        concat = ctx.network.add_concatenation([pad_left.get_output(0), current_tensor])
        concat.axis = 2  # Width axis
        current_tensor = concat.get_output(0)
        current_W += pad_l
    if pad_r > 0:
        zero_data_right = np.zeros((N, current_H, pad_r, C), dtype=np.float32)
        pad_right = ctx.network.add_constant(tuple(zero_data_right.shape), zero_data_right)
        concat = ctx.network.add_concatenation([current_tensor, pad_right.get_output(0)])
        concat.axis = 2  # Width axis
        current_tensor = concat.get_output(0)
        current_W += pad_r

    if pad_s > 0:
        zero_data_left = np.zeros((N, current_H, current_W, pad_s), dtype=np.float32)
        pad_start = ctx.network.add_constant(tuple(zero_data_left.shape), zero_data_left)
        concat = ctx.network.add_concatenation([pad_start.get_output(0), current_tensor])
        concat.axis = 3  # Width axis
        current_tensor = concat.get_output(0)

    if pad_e > 0:
        zero_data_right = np.zeros((N, current_H, current_W, pad_e), dtype=np.float32)
        pad_end = ctx.network.add_constant(tuple(zero_data_right.shape), zero_data_right)
        concat = ctx.network.add_concatenation([current_tensor, pad_end.get_output(0)])
        concat.axis = 3  # Width axis
        current_tensor = concat.get_output(0)

    output._trt = current_tensor

if __name__ == "__main__":
    # 创建一个形状为 (1, 8, 200, 200) 的张量
    x = torch.rand(1, 200, 200, 8).cuda()
    y = torch.rand(1, 200, 200, 8).cuda()
    # 实例化模块
    converter = Padtest().cuda()
    # 调用模块的 forward 方法
    output1 = converter(y)
    # 打印输出张量的形状和前几个元素
    print("Output shape:", output1.shape)
    print("Output tensor:", output1)
    model_trt = torch2trt(converter, [x], max_workspace_size=1 << 30)

    output = model_trt(y)
    # 打印输出张量的形状和前几个元素
    print("Output shape:", output.shape)
    print("Output tensor:", output)

    print(torch.allclose(output1, output, 1e-6))