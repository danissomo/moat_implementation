import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import numpy as np
from einops.layers.torch import Rearrange

class MHSA(nn.Module):
  def __init__(self,
         emb_dim,
         kqv_dim,
         num_heads=1):
    super(MHSA, self).__init__()
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

class MOAT_block(nn.Module):
    def __init__(self, classes, img_size = 32 , patch_size = 4,  c = 12) -> None:
        super().__init__()
        
        self.preconv = nn.Conv2d(3, c, 16, padding=7,  bias=False, stride=4)
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
        img_size_after = round( (img_size+ 2*7- 16- 1)/4 + 1)
        self.layer_norm = nn.LayerNorm(img_size_after)
        patch_dim = c * patch_size ** 2
        emb_size = patch_dim
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
            nn.Linear(patch_dim, emb_size),
        )
        self.self_attn = MHSA(emb_size, 4, 4)
        self.to_class = nn.Sequential(
            Rearrange('b c h w ->  b (c h w)'), 
            nn.Linear(64*64*c, 64*64*c),
            nn.GELU(),
            nn.Linear(64*64*c*2, classes),
            nn.GELU()
        )

    def forward(self, x):
        x = self.preconv(x)
        out1 = self.first_stage(x)
        out1 = x + out1
        
        normed =  self.layer_norm(out1)
        embeded = self.to_patch_embedding(normed)
        
        attended = self.self_attn.forward(embeded)
        attended = attended.view(out1.shape)
        out = attended+out1
        rs = self.to_class(out)
        return rs


class CustomMOAT_torch(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forwarf(self, x):
        pass


