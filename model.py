from torch import nn
import torch
import math


class ConvBranch(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3,
                      padding=1, groups=hidden_dim),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5,
                      padding=2, groups=hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, hidden, seq]
        x = self.convs(x)
        x = x.permute(0, 2, 1)  # [batch, seq, hidden]
        return self.norm(x)


class EnhancedTransformerModel(nn.Module):
    def __init__(self, seq_len, args):
        super().__init__()
        self.input_dim = 6
        self.hidden_dim = args["hidden_dim"]
        self.n_layers = args["n_layers"]
        self.n_heads = args["n_heads"]
        self.add_mean_evec = args.get("mean_evec", False)

        # 嵌入层
        self.embedding = nn.Linear(self.input_dim, self.hidden_dim)
        self.positional_encoding = PositionalEncoding(self.hidden_dim, max_len=seq_len)

        # Transformer编码层
        self.transformer_encoder = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=self.hidden_dim,
                nhead=self.n_heads,
                dim_feedforward=self.hidden_dim * 4,
                dropout=0.1
            ) for _ in range(self.n_layers)
        ])

        # 新增 Hyena 层
        self.hyena = HyenaOperator(
            dim=self.hidden_dim,
            order=2,
            filter_dim=64,
            seq_len=seq_len
        )

        # 卷积分支
        self.conv_branch = ConvBranch(self.hidden_dim)

        # 分类器（根据是否拼接均值特征调整维度）
        if self.add_mean_evec:
            self.classifier = nn.Linear(self.hidden_dim * 3 + 1, 2)
        else:
            self.classifier = nn.Linear(self.hidden_dim * 3, 2)

    def forward(self, inputs, mean_evec=None):
        x = self.embedding(inputs)  # [batch, seq_len, hidden_dim]

        # === 并行计算三个分支 ===
        # ConvBranch 分支（新增全局平均池化）
        x_conv = self.conv_branch(x).mean(dim=1)  # [batch, hidden_dim]

        # Transformer 分支（保持不变）
        x_trans = x.permute(1, 0, 2)  # [seq_len, batch, hidden_dim]
        x_trans = self.positional_encoding(x_trans)
        for layer in self.transformer_encoder:
            x_trans = layer(x_trans)
        x_trans = x_trans[-1]  # 取最后一个时间步 [batch, hidden_dim]

        # Hyena 分支（保持不变）
        x_hyena = self.hyena(x)  # [batch, seq_len, hidden_dim]
        x_hyena = x_hyena.mean(dim=1)  # [batch, hidden_dim]

        # === 特征合并 ===
        x = torch.cat([x_trans, x_hyena, x_conv], dim=1)  # [batch, hidden_dim * 3]

        # === 可选：添加均值特征 ===
        if self.add_mean_evec and mean_evec is not None:
            x = torch.cat([x, mean_evec.unsqueeze(1)], dim=1)  # [batch, hidden_dim * 3 + 1]

        return self.classifier(x)  # 输出分类结果


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(1))

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


# ==== 在 model.py 末尾添加 HyenaOperator 类 ====
class HyenaOperator(nn.Module):
    def __init__(self, dim, order=2, filter_dim=64, seq_len=100):
        super().__init__()
        self.dim = dim
        self.seq_len = seq_len
        self.fft_size = seq_len // 2 + 1  # 计算FFT后的频点数

        # 短卷积分支
        self.short_conv = nn.Conv1d(
            in_channels=dim,
            out_channels=dim,
            kernel_size=3,
            padding=1,
            groups=dim  # 深度可分离卷积
        )

        # 长卷积分支（频域滤波）
        self.filter = nn.Parameter(torch.randn(filter_dim))
        self.filter_proj = nn.Sequential(
            nn.Linear(filter_dim, self.fft_size),
            nn.ReLU()
        )

        # 门控机制
        self.gate = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GLU(dim=-1)
        )

        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [batch, seq_len, dim]
        residual = x

        # === 短卷积分支 ===
        x_short = self.short_conv(x.permute(0, 2, 1)).permute(0, 2, 1)

        # === 长卷积分支 ===
        # 1. 计算FFT
        x_fft = torch.fft.rfft(x, dim=1)  # [batch, fft_size, dim]

        # 2. 生成滤波器（关键修正）
        filt = self.filter_proj(self.filter)  # [fft_size]
        filt = filt.view(1, self.fft_size, 1)  # [1, fft_size, 1] 准备广播

        # 3. 频域相乘（广播机制）
        x_fft = x_fft * filt  # [B, L, D] * [1, L, 1] => [B, L, D]

        # 4. 逆FFT
        x_long = torch.fft.irfft(x_fft, n=self.seq_len, dim=1)

        # === 合并分支 ===
        x = x_short + x_long

        # === 门控机制 ===
        x = x * self.gate(x)

        # === 残差连接 + 归一化 ===
        return self.norm(x + residual)