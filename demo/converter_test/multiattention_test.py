from typing import Optional

import torch
from torch import nn
from torch.nn import functional as F
import  math
from torch import nn, Tensor
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
        self.q_proj.weight.data = q_weight.clone()
        self.k_proj.weight.data = k_weight.clone()
        self.v_proj.weight.data = v_weight.clone()
        self.q_proj.bias.data = q_bias.clone()
        self.k_proj.bias.data = k_bias.clone()
        self.v_proj.bias.data = v_bias.clone()

        self.proj.weight.data = torch_tgt_module.out_proj.weight.data.clone()
        self.proj.bias.data = torch_tgt_module.out_proj.bias.data.clone()

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
        self.export = False

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)

        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]

        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)


        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)

    def set_export(self):
        self.export = True
        batch_masked_attention = PlainMultiHeadAttention(self.d_model, self.nhead)
        batch_masked_attention.set_parameters(self.self_attn)
        self.self_attn = batch_masked_attention


class AttentionMaskConverter(nn.Module):
    def __init__(self):
        super(AttentionMaskConverter, self).__init__()
        self.num_layers = 16
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=256,
                    nhead=8,
                    dropout=0.0,
                    normalize_before=False,
                )
            )
        for i in range(self.num_layers):
            self.transformer_self_attention_layers[i].set_export()


    def forward(self, x,y):
        for i in range(self.num_layers):
            x = self.transformer_self_attention_layers[i]( x, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=y )
        return  x

model = AttentionMaskConverter().cuda()

x = torch.rand(100, 1, 256).cuda()
y = torch.rand(100, 1, 256).cuda()
output1 = model(x,y)

model_trt = torch2trt(model, [x,y], max_workspace_size=1 << 30)
# 调用模块的 forward 方法

# 打印输出张量的形状和前几个元素
print("Output shape:", output1.shape)
print("Output tensor:", output1)
x = torch.rand(100, 1, 256).cuda()
y = torch.rand(100, 1, 256).cuda()
output1 = model(x,y)
output = model_trt(x,y)
print(output.shape)
# 打印输出张量的形状和前几个元素
print("Output shape:", output.shape)
print("Output tensor:", output)
print((torch.max(torch.abs(output - output1))))

