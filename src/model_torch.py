import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import numpy as np
from einops.layers.torch import Rearrange
import math

class MHSA(nn.Module):
  def __init__(self,
         emb_dim,
         kqv_dim,
         num_heads=1):
    super().__init__()
    self.emb_dim = emb_dim
    self.kqv_dim = kqv_dim
    self.num_heads = num_heads

    self.w_k = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_q = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_v = nn.Linear(emb_dim, kqv_dim * num_heads, bias=False)
    self.w_out = nn.Linear(kqv_dim * num_heads, emb_dim)

  def forward(self, x):

    b, t, _ = x.shape
    e = self.kqv_dim
    h = self.num_heads
    keys = self.w_k(x).view(b, t, h, e)
    values = self.w_v(x).view(b, t, h, e)
    queries = self.w_q(x).view(b, t, h, e)

    keys = keys.transpose(2, 1)
    queries = queries.transpose(2, 1)
    values = values.transpose(2, 1)

    dot = queries @ keys.transpose(3, 2)
    dot = dot / np.sqrt(e)
    dot = F.softmax(dot, dim=3)

    out = dot @ values
    out = out.transpose(1,2).contiguous().view(b, t, h * e)
    out = self.w_out(out)
    return out

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class MOAT_block(nn.Module):
    def __init__(self, classes, img_size = 32 , patch_size = 16,  c =32, dim_head = 128, heads = 4) -> None:
        super().__init__()
    
        self.first_stage = nn.Sequential(
            nn.BatchNorm2d(c),
            nn.Conv2d(c, 4*c, kernel_size=(1, 1)),
            nn.BatchNorm2d(4*c),
            nn.GELU(), 
            nn.Conv2d(4*c, 3*4*c, kernel_size=3, groups=4*c, padding=1),
            nn.BatchNorm2d(3*4*c),
            nn.GELU(), 
            nn.Conv2d(3*4*c, c, (1,1))
        )
        
        img_size_after = img_size
        self.layer_norm = nn.LayerNorm((img_size_after, img_size_after))
        patch_dim = c * patch_size ** 2
        emb_size = patch_dim
        # self.mha_torch = nn.MultiheadAttention(emb_size, 4, 0.2)
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, emb_size),
        )
        num_patches = (img_size// patch_size) * (img_size // patch_size)
        self.pos_embedding = PositionalEncoding(patch_dim)
        
        self.self_attn = MHSA(emb_size, dim_head, heads)
        
        

    def forward(self, x):
        # x = self.to_stem(x)
        out = x + self.first_stage(x)
        
        patches =  self.layer_norm(out)
        
        patches = self.to_patch_embedding(patches)
        b, n, _ = patches.shape
        patches = self.pos_embedding(patches)
        # patches,_ = self.mha_torch(patches, patches, patches, need_weights=False)
        
        patches = self.self_attn.forward(patches)
        patches = patches.view(out.shape)
        out = patches+out

        return out

def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
            from math import floor
            if type(kernel_size) is not tuple:
                kernel_size = (kernel_size, kernel_size)
            h = floor( ((h_w[0] + (2 * pad) - ( dilation * (kernel_size[0] - 1) ) - 1 )/ stride) + 1)
            w = floor( ((h_w[1] + (2 * pad) - ( dilation * (kernel_size[1] - 1) ) - 1 )/ stride) + 1)
            return h, w

class CustomMOAT_torch(nn.Module):
    def __init__(self, classes, img_size = 32 , patch_size = 16,  c =32, dim_head = 128, heads = 4, dim_mlp = 100) -> None:
        super().__init__()
        pad, stri, k = 7, 4, 16
        self.to_stem = nn.Sequential(
             nn.Conv2d(3, c, k, padding=pad,  bias=False, stride=stri),
             nn.GELU(),
        )
        
        img_size_after = round( (img_size+ 2*pad- 1*(k-1) - 1)/stri + 1)
        self.moat_block_1 = MOAT_block(classes, img_size_after, patch_size, c, dim_head, heads)
        # self.dec_channels = nn.Conv2d(c, c*2, 8, bias=False, stride=2)
        # end_size = conv_output_shape((img_size_after, img_size_after), 8, stride=2)
        # self.moat_block_2 = MOAT_block(classes, end_size[0], 3, c*2)
        end_size = (img_size_after, img_size_after)
        self.to_class = nn.Sequential(
            Rearrange('b c h w ->  b (c h w)'), 
            nn.Linear(end_size[0]*end_size[1]*c, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, dim_mlp),
            nn.GELU(),
            nn.Linear(dim_mlp, classes),
        )
        # добавление скрытого слоя сушественно ускорило обучение и точность
        # но точность после 0.75 на тесте повышается незначительно
        # в то время как на трейне растет
        
        
    def forward(self, x):
        x = self.to_stem(x)
        x = self.moat_block_1(x)
        # x = self.dec_channels(x)
        # x = self.moat_block_2(x)
        x = self.to_class(x)
        return x


