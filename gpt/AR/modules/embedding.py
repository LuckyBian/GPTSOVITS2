# modified from https://github.com/lifeiteng/vall-e/blob/main/valle/modules/embedding.py
import math

import torch
from torch import nn


class TokenEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.dropout = torch.nn.Dropout(p=dropout)

        # 查看x索引的范围，决定这层的大小。比如x索引范围是0-9，这个范围要大于等于10.
        # embedding_dim 相当于把 x中的一个索引转换成的维度大小
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

    @property
    def weight(self) -> torch.Tensor:
        return self.word_embeddings.weight

    def embedding(self, index: int) -> torch.Tensor:
        return self.word_embeddings.weight[index : index + 1]

    def forward(self, x: torch.Tensor):
        x = self.word_embeddings(x)
        x = self.dropout(x)
        return x


class SinePositionalEmbedding(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        dropout: float = 0.0,
        scale: bool = False,
        alpha: bool = False,
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.x_scale = math.sqrt(embedding_dim) if scale else 1.0
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=alpha)
        self.dropout = torch.nn.Dropout(p=dropout)

        self.reverse = False
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, 4000))

    def extend_pe(self, x):
        """Reset the positional encodings."""
        if self.pe is not None:
            if self.pe.size(1) >= x.size(1):
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
            
        # 创建一个变量，值都是0，长度和x一样，维度和x一样
        pe = torch.zeros(x.size(1), self.embedding_dim)

        if self.reverse:
            position = torch.arange(
                x.size(1) - 1, -1, -1.0, dtype=torch.float32
            ).unsqueeze(1)

        else:
            # 根据x序列长度，新建了一个序列，标记位置，提升维度
            position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)

        # torch.arange(0, self.embedding_dim, 2, dtype=torch.float32)： 只在偶数位置生成值
        # -(math.log(10000.0) / self.embedding_dim)： 定义波长范围然后取对数进行缩放。
        # 确保随着嵌入维度的增加，波长的缩放比例得到适当的调整。这保证了无论嵌入向量的大小如何，波长变化都是平滑的。
        # 再次进行缩放，比较不同位置的细微差别
        div_term = torch.exp(
            torch.arange(0, self.embedding_dim, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.embedding_dim)
        )

        # 索引偶数位，0开始步长为2
        pe[:, 0::2] = torch.sin(position * div_term)

        # 索引奇数位，1开始步长为2
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.pe = pe.to(device=x.device, dtype=x.dtype).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # 得到pe，就是得到x的位置编码
        self.extend_pe(x)

        # x是三维的，不变
        output = x.unsqueeze(-1) if x.ndim == 2 else x

        # 将原始信息与位置信息相互结合
        output = output * self.x_scale + self.alpha * self.pe[:, : x.size(1)]

        return self.dropout(output)
