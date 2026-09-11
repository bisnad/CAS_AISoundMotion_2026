import torch
from torch import nn
import math
import torch.nn.functional as nnF

config = {
    "input_dim": 308, 
    "embed_dim": 512,
    "num_heads": 8,
    "num_layers": 6,
    "ff_dim": 2048,
    "dropout": 0.1,
    "pos_encoding_max_length": 64,
    "num_mixtures": 20,
    "device": "cuda",
    "weights_path": ""
}

class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        seq_len = token_embedding.size(1)
        pe = self.pos_encoding[:, :seq_len, :]
        return self.dropout(token_embedding + pe)

class MDNLayer(nn.Module):
    def __init__(self, hidden_dim, out_dim, num_mixtures=5):
        super().__init__()
        self.num_mixtures = num_mixtures
        self.out_dim = out_dim
        self.pi_head = nn.Linear(hidden_dim, num_mixtures)
        self.mu_head = nn.Linear(hidden_dim, num_mixtures * out_dim)
        self.sigma_head = nn.Linear(hidden_dim, num_mixtures * out_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        log_pi = nnF.log_softmax(self.pi_head(x), dim=-1)
        mu = self.mu_head(x).view(batch_size, seq_len, self.num_mixtures, self.out_dim)
        sigma = self.sigma_head(x).view(batch_size, seq_len, self.num_mixtures, self.out_dim)
        sigma = nnF.elu(sigma) + 1.0 + 1e-6
        return log_pi, mu, sigma

class TransformerDecoderMDN(nn.Module):
    def __init__(self, motion_dim, embed_dim, num_heads, num_decoder_layers, ff_dim, dropout_p, pos_encoding_max_length, num_mixtures=5):
        super().__init__()
        self.embed_dim = embed_dim
        self.motion2embed = nn.Linear(motion_dim, embed_dim)
        self.positional_encoder = PositionalEncoding(dim_model=embed_dim, dropout_p=dropout_p, max_len=pos_encoding_max_length * 2)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, dropout=dropout_p, activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_decoder_layers)
        self.mdn = MDNLayer(embed_dim, motion_dim, num_mixtures)

    def forward(self, motion_data, return_sequence=False):
        seq_len = motion_data.size(1)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=motion_data.device)
        motion_embedded = self.motion2embed(motion_data) * math.sqrt(self.embed_dim)
        motion_embedded = self.positional_encoder(motion_embedded)
        decoder_out = self.transformer(motion_embedded, mask=tgt_mask, is_causal=True)
        log_pi, mu, sigma = self.mdn(decoder_out)

        if return_sequence:
            return log_pi, mu, sigma
        else:
            return log_pi[:, -1:, :], mu[:, -1:, :, :], sigma[:, -1:, :, :]

def createModel(config):
    decoder = TransformerDecoderMDN(
        motion_dim=config["input_dim"],
        embed_dim=config["embed_dim"],
        num_heads=config["num_heads"],
        num_decoder_layers=config["num_layers"],
        ff_dim=config["ff_dim"],
        dropout_p=config["dropout"],
        pos_encoding_max_length=config["pos_encoding_max_length"],
        num_mixtures=config["num_mixtures"]
    ).to(config["device"])

    if config["weights_path"] != "":
        if config["device"] == 'cuda':
            decoder.load_state_dict(torch.load(config["weights_path"]))
        else:
            decoder.load_state_dict(torch.load(config["weights_path"], map_location=torch.device(config["device"])))

    decoder.eval()
    return decoder