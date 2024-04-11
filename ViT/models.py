import torch
import torch.nn as nn
from einops import repeat
from einops.layers.torch import Rearrange

#Pathcing
class Patching(nn.Module):
    def __init__(self, patch_size):
        """
        patch_size(int) : パッチの縦の長さ（=横の長さ）
        """
        super().__init__()
        self.net = Rearrange("b c (h ph) (w pw) -> b (h w) (ph pw c)", ph = patch_size, pw = patch_size)
    
    def forward(self, x):
        """
        x (torch.Tensor) : 画像データ
            x.shape = torch.size([batch_size, channels, height, width])
        """
        x = self.net(x)
        return x

#各パッチをベクトルに変換
class LinearProjection(nn.Module):
    def __init__(self, patch_dim, dim):
        super().__init__()
        """
        - patch_dim (int) : 一枚当たりのパッチの次元(c * pathch_size^2)
        - dim (int) : パッチが変換されたベクトルの次元
        """
        self.net = nn.Linear(patch_dim, dim)
    
    def forward(self, x):
        """
         - x (torch.Tensor)
          - x.shape : (batch_size, n_patch, patch_dim)
        """
        return self.net(x)
    