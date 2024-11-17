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
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        if not self._qkv_same_embed_dim:
            assert NotImplementedError
        else:
            self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        #self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def init_weights(self):
        pass

    def scaled_dot_product_attention(self, query, key, value, attn_mask=None, dropout_p=0.0,
                                     is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
        L, S = query.size(-2), key.size(-2)
        scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale

        if enable_gqa:
            key = key.repeat_interleave(query.size(-3) // key.size(-3), -3)
            value = value.repeat_interleave(query.size(-3) // value.size(-3), -3)

        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        if attn_mask is not None:
            attn_weight += attn_mask
        attn_weight = F.softmax(attn_weight, dim=-1)
        if dropout_p > 0:
            attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
        return attn_weight @ value
    def attn_mask_process(self, attn_mask):
        pass

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
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape


        qkv = self.qkv(query)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)

        attn_output = self.scaled_dot_product_attention(q, k, v, attn_mask, dropout_p, is_causal)
        attn_output = attn_output.permute(2, 0, 1, 3).contiguous().view(bsz * tgt_len, embed_dim)
        attn_output = self.proj(attn_output)
        attn_output = attn_output.view(tgt_len, bsz, attn_output.size(1))
        if self.batch_first and is_batched:
            return attn_output.transpose(1, 0), None
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
        self.qkv.weight.data = torch_tgt_module.in_proj_weight.data
        self.qkv.bias.data = torch_tgt_module.in_proj_bias.data
        self.proj.weight.data = torch_tgt_module.out_proj.weight.data
        self.proj.bias.data = torch_tgt_module.out_proj.bias.data


if __name__ == '__main__':
    self_attn = True
    # Function to compare outputs
    def compare_outputs():
        embed_dim = 256
        num_heads = 8

        attn1 = nn.MultiheadAttention(embed_dim, num_heads,dropout=0.1).cuda()
        attn2 = PlainMultiHeadAttention(embed_dim, num_heads,dropout=0.1).cuda()
        attn2.set_parameters(attn1)

        attn2.eval()
        attn1.eval()


        batch_size = 1
        seq_len = 100
        attn_mask = torch.randint(0, batch_size, (seq_len, seq_len)).bool().cuda()
        query = torch.rand(seq_len, batch_size, embed_dim).cuda()
        if self_attn:
            key = query.clone()
            value = query.clone()
        else:
            key = torch.rand(seq_len, batch_size, embed_dim)
            value = torch.rand(seq_len, batch_size, embed_dim)

        output1, _ = attn1(query, key, value,attn_mask =attn_mask)
        output2, _ = attn2(query, key, value,attn_mask =attn_mask)

        #output1, _ = attn1(query, key, value)
        #output2, _ = attn2(query, key, value)

        print("Output from nn.MultiheadAttention:")
        print(output1)

        print("Output from PlainMultiHeadAttention:")
        print(output2)

        print("Outputs are close:", torch.allclose(output1, output2, atol=1e-6))

        model_trt = torch2trt(attn2, [query+1, key+1, value+1], max_workspace_size=1 << 30)
        output2, _ = attn2(query+1, key+1, value+1, attn_mask=attn_mask)
        out, _ = model_trt(query+1, key+1, value+1, attn_mask)


        print("Outputs are close:", torch.allclose(out, output2, atol=1e-6))
        print((torch.max(torch.abs(out - output2))))

    compare_outputs()

