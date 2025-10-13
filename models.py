import torch
import torch.nn as nn
from mamba_ssm import Mamba 
import math

class CNN1D(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(CNN1D, self).__init__()
        defaults = {
            "n_filters": 32,
            "kernel_size": 3,
            "stride": 1,
            "padding":1,
            "n_layers": 3,
            "dropout": 0.3,
            "pooling": "max",
            "pool_kernel": 2,
            "pool_stride": 1,
            "batch_norm": False,
            "hidden_dim_fc": 128
        }
        # 用自定义参数覆盖默认值
        cfg = {**defaults, **kwargs}
        
        layers = []
        in_channels = 1
        current_dim = input_dim

        for _ in range(cfg["n_layers"]):
            layers.append(nn.Conv1d(in_channels=in_channels, 
                                    out_channels=cfg["n_filters"], 
                                    kernel_size=cfg["kernel_size"], 
                                    stride=cfg["stride"], 
                                    padding=cfg["padding"]))
            if cfg["batch_norm"]:
                layers.append(nn.BatchNorm1d(cfg["n_filters"]))
            
            layers.append(nn.ReLU())

            if cfg["pooling"] == "max":
                layers.append(nn.MaxPool1d(kernel_size=cfg["pool_kernel"], stride=cfg["pool_stride"]))
                current_dim = (current_dim + 2 * cfg["padding"] - cfg["kernel_size"]) // cfg["stride"] + 1
                current_dim = (current_dim - cfg["pool_kernel"]) // (cfg["pool_stride"]) + 1
            elif cfg["pooling"] == "avg":
                layers.append(nn.AvgPool1d(kernel_size=cfg["pool_kernel"], stride=cfg["pool_stride"]))
                current_dim = (current_dim + 2 * cfg["padding"] - cfg["kernel_size"]) // cfg["stride"] + 1
                current_dim = (current_dim - cfg["pool_kernel"]) // (cfg["pool_stride"]) + 1
            else:
                # 没有池化
                current_dim = (current_dim + 2 * cfg["padding"] - cfg["kernel_size"]) // cfg["stride"] + 1
            in_channels = cfg["n_filters"]

        self.conv = nn.Sequential(*layers)
        self.dropout = nn.Dropout(cfg["dropout"])

        fc_in = cfg["n_filters"] * current_dim
        if cfg["hidden_dim_fc"]:
            self.fc = nn.Sequential(
                nn.Linear(fc_in, cfg["hidden_dim_fc"]),
                nn.ReLU(),
                nn.Linear(cfg["hidden_dim_fc"], 1)
            )
        else:
            self.fc = nn.Linear(fc_in, 1)
        
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.squeeze(-1)
    
class LSTM1D(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(LSTM1D, self).__init__()
        defaults = {
            "hidden_size": 64,
            "bias": True,
            "n_layers": 3,
            "dropout": 0.3,
            "batch_first": True
        }
        cfg = {**defaults, **kwargs}

        self.lstm = nn.LSTM(input_dim, 
                            hidden_size=cfg["hidden_size"], 
                            num_layers=cfg["n_layers"],
                            dropout=cfg["dropout"] if cfg["n_layers"] > 1 else 0.0,
                            bias=cfg["bias"],
                            batch_first=cfg["batch_first"])
        self.norm = nn.LayerNorm(cfg["hidden_size"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(cfg["hidden_size"], 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)  # [B, L] -> [B, L, 1]
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])  # 取最后一个时间步
        out = self.dropout(out)
        out = self.fc(out)
        return out.view(-1)  # 确保输出是 [B]


class GRU1D(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(GRU1D, self).__init__()
        defaults = {
            "hidden_size": 64,
            "bias": True,
            "n_layers": 3,
            "dropout": 0.3,
            "batch_first": True
        }
        cfg = {**defaults, **kwargs}

        self.gru = nn.GRU(input_dim, cfg["hidden_size"], 
                          num_layers=cfg["n_layers"],
                          dropout=cfg["dropout"] if cfg["n_layers"] > 1 else 0.0,
                          bias=cfg["bias"],
                          batch_first=cfg["batch_first"])
        self.norm = nn.LayerNorm(cfg["hidden_size"])
        self.dropout = nn.Dropout(cfg["dropout"])
        self.fc = nn.Linear(cfg["hidden_size"], 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        out, _ = self.gru(x)
        out = self.norm(out[:, -1, :])
        out = self.dropout(out)
        out = self.fc(out)
        return out.view(-1)
    
    
class TransformerEncoderLayerWithAttn(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.attn_weights = None  

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2, attn = self.self_attn(src, src, src, attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask, need_weights=True)
        self.attn_weights = attn 
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src
    

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        pe = self.pe[:x.size(1), :].unsqueeze(0)  # [1, seq_len, d_model]
        x = x + pe.to(x.device) 
        return x

class Transformer1D(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super(Transformer1D, self).__init__()
        defaults = {
            "hidden_dim": 64,
            "n_layers": 3,
            "dropout": 0.3,
            "nhead": 4,
            "max_seq_len": 1023,  
        }
        cfg = {**defaults, **kwargs}
        
        self.embedding = nn.Linear(input_dim, cfg["hidden_dim"])

        self.pos_encoder = PositionalEncoding(cfg["hidden_dim"], cfg["max_seq_len"])
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithAttn(
                d_model=cfg["hidden_dim"],
                nhead=cfg["nhead"],
                dim_feedforward=cfg["hidden_dim"] * 2,
                dropout=cfg["dropout"],
                batch_first=True
            ) for _ in range(cfg["n_layers"])
        ])
        
        self.fc = nn.Linear(cfg["hidden_dim"], 1)
    
    def forward(self, x):
        x = x.unsqueeze(-1)
        x = self.embedding(x)
        x = self.pos_encoder(x)
        for layer in self.encoder_layers:
            x = layer(x)
        x = self.fc(x[:, -1, :]).squeeze(-1)
        return x   


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        # x: (B, L, C=d_model)
        y = self.mamba(self.norm(x))
        return x + self.drop(y)  # Residual

class Mamba1D(nn.Module):
    def __init__(self, **kwargs):
        """
        Input convention:
        Original X: (B, L) -- L= Feature dimension
        In this model, the channel is first projected from 1 to d_model internally, and then MambaBlock is stacked.
        """
        super().__init__()
        defaults = {
            "d_model": 64,     
            "d_state": 16,
            "d_conv": 4,
            "expand": 2,
            "n_layers": 3,   
            "dropout": 0.2,
            "pooling": "avg",
            "head_hidden": 64    
        }
        cfg = {**defaults, **kwargs}
        self.pooling = cfg["pooling"]
        d_model = cfg["d_model"]

        self.input_proj = nn.Linear(1, d_model)
        blocks = []
        for _ in range(cfg["n_layers"]):
            blocks.append(MambaBlock(d_model, cfg["d_state"], cfg["d_conv"], cfg["expand"], cfg["dropout"]))
        self.blocks = nn.Sequential(*blocks)
        self.dropout = nn.Dropout(cfg["dropout"])
        self.head = nn.Sequential(
            nn.Linear(d_model, cfg["head_hidden"]),
            nn.ReLU(),
            nn.Dropout(cfg["dropout"]),
            nn.Linear(cfg["head_hidden"], 1)
        )
        nn.init.xavier_uniform_(self.head[0].weight); nn.init.zeros_(self.head[0].bias)
        nn.init.xavier_uniform_(self.head[3].weight); nn.init.zeros_(self.head[3].bias)

    def forward(self, x):
        x = x.unsqueeze(-1)          
        x = self.input_proj(x)         
        x = self.blocks(x)            
        if self.pooling == "avg":
            x = x.mean(dim=1)         
        elif self.pooling == "max":
            x = x.max(dim=1).values    
        elif self.pooling == "last":
            x = x[:, -1, :]            
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling}")
        x = self.dropout(x)
        x = self.head(x)              
        return x.squeeze(-1)          