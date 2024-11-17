import torch.nn as nn
import torch
import torch.nn.functional as F
import math

from torch2trt.torch2trt import *


class PlainMultiHeadAttention(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            num_heads=16,
            dropout=0.,
            bias=True,
            kdim=None,
            vdim=None,
            batch_first=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        # 统一使用独立的投影层
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(self.kdim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(self.vdim, embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
                                     is_causal=False, scale=None) -> torch.Tensor:
        # query: [B, H, L, D]
        # key:   [B, H, S, D]
        # value: [B, H, S, D]
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

        # [B, H, L, S]
        attn_weight = query @ key.transpose(-2, -1) * scale_factor

        if is_causal:
            L, S = query.size(-2), key.size(-2)
            causal_mask = torch.ones(L, S, dtype=torch.bool, device=query.device).triu_(1)
            attn_weight.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attn_mask is not None:
            attn_weight = attn_weight + attn_mask

        attn_weight = F.softmax(attn_weight, dim=-1)

        if dropout_p > 0:
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)

        # [B, H, L, D]
        return attn_weight @ value

    def forward(
            self,
            query,
            key,
            value,
            attn_mask=None,
            key_padding_mask=None,
            need_weights=True,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")

        is_batched = query.dim() == 3

        if self.batch_first and is_batched:
            # [B, L, E] -> [L, B, E]
            query, key, value = [x.transpose(1, 0) if x is not None else None
                                 for x in (query, key, value)]

        if not is_batched:
            query = query.unsqueeze(1)
            key = key.unsqueeze(1)
            value = value.unsqueeze(1)

        tgt_len, bsz, embed_dim = query.shape
        src_len = key.shape[0]

        # 独立投影
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        # [L/S, B*H, D]
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.contiguous().view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        # [B, H, L/S, D]
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        # # Handle attention mask
        # if attn_mask is not None:
        #     if attn_mask.dim() == 2:
        #         attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
        #     elif attn_mask.dim() == 3:
        #         attn_mask = attn_mask.unsqueeze(1)
        #     # Ensure proper broadcasting
        #     if attn_mask.size(-2) == 1:
        #         attn_mask = attn_mask.expand(-1, -1, tgt_len, -1)
        #     if attn_mask.size(-1) == 1:
        #         attn_mask = attn_mask.expand(-1, -1, -1, src_len)

        dropout_p = self.dropout if self.training else 0.0

        # Compute attention
        attn_output = self.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        )

        # Reshape and project output
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous()
        attn_output = attn_output.view(tgt_len, bsz, embed_dim)
        attn_output = self.proj(attn_output.view(-1, embed_dim)).view(tgt_len, bsz, embed_dim)

        if self.batch_first and is_batched:
            attn_output = attn_output.transpose(1, 0)

        return attn_output, None

    def set_parameters(self, torch_tgt_module):
        assert isinstance(torch_tgt_module, nn.MultiheadAttention)
        assert self.embed_dim == torch_tgt_module.embed_dim
        assert self.batch_first == torch_tgt_module.batch_first
        assert self.dropout == torch_tgt_module.dropout
        assert self.head_dim == torch_tgt_module.head_dim
        assert self.num_heads == torch_tgt_module.num_heads
        assert self.kdim == torch_tgt_module.kdim
        assert self.vdim == torch_tgt_module.vdim

        # 直接从in_proj_weight分割权重
        q_weight, k_weight, v_weight = torch_tgt_module.in_proj_weight.chunk(3)
        q_bias, k_bias, v_bias = torch_tgt_module.in_proj_bias.chunk(3)

        # 设置各投影层的参数
        self.q_proj.weight.data = q_weight
        self.k_proj.weight.data = k_weight
        self.v_proj.weight.data = v_weight
        self.q_proj.bias.data = q_bias
        self.k_proj.bias.data = k_bias
        self.v_proj.bias.data = v_bias

        self.proj.weight.data = torch_tgt_module.out_proj.weight.data
        self.proj.bias.data = torch_tgt_module.out_proj.bias.data

if __name__ == '__main__':
    self_attn = True


    # Function to compare outputs
    def compare_outputs():
        embed_dim = 256
        num_heads = 4

        attn1 = nn.MultiheadAttention(embed_dim, num_heads,dropout=0.1).cuda()
        attn2 = PlainMultiHeadAttention(embed_dim, num_heads,dropout=0.1).cuda()
        attn2.set_parameters(attn1)

        attn2.eval()
        attn1.eval()

        batch_size = 2
        seq_len = 100

        attn_mask = torch.randint(0,2,(seq_len, seq_len)).bool().cuda()

        attn_mask1 = attn_mask
        # 先将 False 位置填充为 -1e8
        #attn_mask = attn_mask.float() * float("-inf") + (1 - attn_mask.float()) * 0
        float_attention_mask = attn_mask.float()
        float_attention_mask[attn_mask] = float('-inf')
        float_attention_mask[~attn_mask] = 0
        attn_mask = float_attention_mask


        attn_mask2 = attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, num_heads, 1, 1)
        query = torch.rand(seq_len, batch_size, embed_dim).cuda()
        if self_attn:
            key = query.clone()
            value = query.clone()
        else:
            k_seq_len = 270
            attn_mask = torch.randint(0, 2, (seq_len, k_seq_len)).bool().cuda()
            attn_mask = attn_mask.float() * 0 + (1 - attn_mask.float()) * -1e8
            attn_mask2 = attn_mask.unsqueeze(0).unsqueeze(1).repeat(batch_size, num_heads, 1, 1)

            key = torch.rand(k_seq_len, batch_size, embed_dim).cuda()
            value = torch.rand(k_seq_len, batch_size, embed_dim).cuda()

        output1, _ = attn1(query, key, value, attn_mask=attn_mask1)
        output2, _ = attn2(query, key, value, attn_mask=attn_mask2)



        #output1, _ = attn1(query, key, value)
        #output2, _ = attn2(query, key, value)

        print("Output from nn.MultiheadAttention:")
        print(output1)

        print("Output from PlainMultiHeadAttention:")
        print(output2)

        print("Outputs are close:", torch.allclose(output1, output2, atol=1e-6))

        # model_trt = torch2trt(attn2, [query, key, value,attn_mask],max_workspace_size=1 << 30)
        # out = model_trt(query, key, value, attn_mask)
        # query = torch.rand(seq_len, batch_size, embed_dim).cuda()
        # if self_attn:
        #     key = query.clone()
        #     value = query.clone()
        # output2, _ = attn2(query, key, value, attn_mask=attn_mask2)
        # out,_ = model_trt(query, key, value, attn_mask)


        print("Outputs are close:", torch.allclose(out, output2, atol=1e-6))

    compare_outputs()

