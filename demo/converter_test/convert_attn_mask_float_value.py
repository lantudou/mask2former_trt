
from torch2trt.torch2trt import *
import torch
import torch.nn as nn
class AttentionMaskConverter(nn.Module):
    def __init__(self):
        super(AttentionMaskConverter, self).__init__()
    def convert_attn_mask_float_value(self, repeated_mask):
        attn_mask = (repeated_mask < 0.5)
        attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

        float_attention_mask = attn_mask.float()
        float_attention_mask[attn_mask] = float('-inf')
        float_attention_mask[~attn_mask] = 0

        return float_attention_mask
    def forward(self, repeated_mask):
        return  self.convert_attn_mask_float_value(repeated_mask)


@tensorrt_converter(AttentionMaskConverter.convert_attn_mask_float_value)
def convert_convert_attn_mask_float_value(ctx):
    module = get_arg(ctx, 'self', pos=0, default=None)
    input_tensor = get_arg(ctx, 'repeated_mask', pos=1, default=None)

    output = ctx.method_return
    # 获取输入张量

    input_trt, threshold_constant_trt = add_missing_trt_tensors(ctx.network, [input_tensor, 0.5])
    input_trt, threshold_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, threshold_constant_trt],
                                                     max(len(input_trt.shape), len(threshold_constant_trt.shape)))

    _, zero_constant_trt = add_missing_trt_tensors(ctx.network, [input_tensor, 0])
    _, zero_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, zero_constant_trt],
                                                     max(len(input_trt.shape), len(zero_constant_trt.shape)))
    _, neg_constant_trt = add_missing_trt_tensors(ctx.network, [input_tensor, float("-inf")])
    _, neg_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, neg_constant_trt],
                                                 max(len(input_trt.shape), len(neg_constant_trt.shape)))

    _, one_constant_trt = add_missing_trt_tensors(ctx.network, [input_tensor, 1.0])
    _, one_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, one_constant_trt],
                                                max(len(input_trt.shape), len(one_constant_trt.shape)))

    # Create shape constant with proper shape for broadcasting
    shape_length = input_trt.shape[-1]
    shape_dims = [1] * len(input_trt.shape)
    shape_dims[-1] = 1  # Match the reduced dimension
    _, shape_constant_trt = add_missing_trt_tensors(ctx.network,
                                                    [input_tensor, float(shape_length)])
    _, shape_constant_trt = broadcast_trt_tensors(ctx.network, [input_trt, shape_constant_trt],
                                                max(len(input_trt.shape), len(shape_constant_trt.shape)))

    # bool compare_less = input < 0.5 ?
    compare_less = ctx.network.add_elementwise(
        input_trt,
        threshold_constant_trt,
        trt.ElementWiseOperation.LESS
    )
    # if compare_less  bool_to_float = 1 else bool_to_float = 0
    bool_to_float = ctx.network.add_select(
        compare_less.get_output(0),
        one_constant_trt,
        zero_constant_trt
    )

    #  reduce_layer = ADD compare_less
    reduce_layer = ctx.network.add_reduce(
        bool_to_float.get_output(0),
        trt.ReduceOperation.SUM,
        (1 << (len(input_trt.shape) - 1)),  # Reduce last dimension
        keep_dims=True  # Keep dimensions to make broadcasting work
    )

    # bool all_true_compare = ( reduce_layer == shape)
    all_true_compare = ctx.network.add_elementwise(
        reduce_layer.get_output(0),
        shape_constant_trt,
        trt.ElementWiseOperation.EQUAL
    )

    # Broadcast the comparison result back to original shape if needed
    if len(all_true_compare.get_output(0).shape) != len(input_trt.shape):
        shuffle = ctx.network.add_shuffle(all_true_compare.get_output(0))
        shuffle.reshape_dims = input_trt.shape
        broadcast_compare = shuffle.get_output(0)
    else:
        broadcast_compare = all_true_compare.get_output(0)

    # if all_true_compare updated_compare = 0  else updated_compare = bool_to_float
    updated_compare = ctx.network.add_select(
        broadcast_compare,
        zero_constant_trt,
        bool_to_float.get_output(0)
    )

    # if bool_to_float > 0  bool_to_float = TRUE else bool_to_float = FLASE
    float_to_bool = ctx.network.add_elementwise(
        updated_compare.get_output(0),
        zero_constant_trt,
        trt.ElementWiseOperation.GREATER
    )

    final_mask = ctx.network.add_select(
        float_to_bool.get_output(0),
        neg_constant_trt,
        zero_constant_trt
    )

    output._trt = final_mask.get_output(0)




# 测试代码
if __name__ == "__main__":
    # 创建一个形状为 (1, 8, 200, 200) 的张量
    x = torch.rand(1, 8, 200, 200).cuda()
    y = torch.rand(1, 8, 200, 200).cuda()
    y[:, 1:5, :, :] = 1
    # 实例化模块
    converter = AttentionMaskConverter().cuda()
    model_trt = torch2trt(converter, [x], max_workspace_size=1 << 30)
    # 调用模块的 forward 方法
    output1 = converter(y)
    # 打印输出张量的形状和前几个元素
    print("Output shape:", output1.shape)
    print("Output tensor:", output1)
    output = model_trt(y)
    # 打印输出张量的形状和前几个元素
    print("Output shape:", output.shape)
    print("Output tensor:", output)
    output.masked_fill_(output == -1.0000e+38, float("-inf"))
    print(torch.allclose(output1,output, 1e-6))

