import torch
import torch.nn as nn
from torch2trt.torch2trt import *

class MultiheadAttentionWrapper(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiheadAttentionWrapper, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads)
    
    def forward(self, query, key, value, attn_mask=None):
        # 确保输入的形状为 (seq_len, batch_size, embed_dim)
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        
        attn_mask = (attn_mask < 0.5).bool()
        print(attn_mask.shape)
        #attn_mask = final_mask
        #attn_mask = attn_mask.detach()
        attn_mask[0:2, 0:2] = True
        #print(attn_mask)
        # 前向传播
        attn_output, attn_output_weights = self.multihead_attn(query, key, value, attn_mask=attn_mask)
        
        # 将输出形状转换回 (batch_size, seq_len, embed_dim)
        attn_output = attn_output.transpose(0, 1)
        
        return attn_output, attn_output_weights
# 定义输入张量
# 假设我们有一个序列长度为 5，每个元素的维度为 4
# 批量大小为 1
batch_size = 1
seq_len = 100
embed_dim = 256
num_heads = 4
# 输入张量 (batch_size, seq_len, embed_dim)
query = torch.randn(batch_size, seq_len, embed_dim).cuda()
key = torch.randn(batch_size, seq_len, embed_dim).cuda()
value = torch.randn(batch_size, seq_len, embed_dim).cuda()



# 创建 attn_mask
# 假设我们想要屏蔽某些位置
# 这里我们屏蔽第一个位置到第二个位置之间的注意力
attn_mask = torch.randn(batch_size*num_heads, seq_len, seq_len).cuda()

# 创建 MultiheadAttentionWrapper 模块
multihead_attn_wrapper = MultiheadAttentionWrapper(embed_dim=embed_dim, num_heads=num_heads)
multihead_attn_wrapper = multihead_attn_wrapper.cuda().eval()

model_trt = torch2trt(multihead_attn_wrapper, [query, key, value, attn_mask])
    # 前向传播
attn_output, attn_output_weights = multihead_attn_wrapper(query, key, value, attn_mask=attn_mask)
attn_output_trt, attn_output_weights_trt = model_trt(query, key, value,attn_mask)
print(torch.max(torch.abs(attn_output - attn_output_trt )))
print(torch.max(torch.abs(attn_output_weights - attn_output_weights_trt)))

# 输出结果
print("Attention Output:", attn_output)
print("Attention Weights:", attn_output_weights)