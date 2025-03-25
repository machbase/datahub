import torch
import torch.nn as nn

class Transformer_Encoder(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(Transformer_Encoder, self).__init__()
        
        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))

        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src):
        # src: (batch_size, seq_len, input_dim)
        seq_len = src.size(1)
        src = self.input_embedding(src) + self.positional_encoding[:, :seq_len, :]
        memory = self.transformer_encoder(src)
        out = self.fc_out(memory)
        
        return out

class Transformer(nn.Module):
    def __init__(self, input_dim, model_dim, num_heads, num_layers, output_dim, dropout=0.1):
        super(Transformer, self).__init__()

        self.model_dim = model_dim
        self.input_embedding = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.zeros(1, 1000, model_dim))

        # Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=dropout
        )
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers
        )

        # Decoder
        self.decoder_layer = nn.TransformerDecoderLayer(
            d_model=model_dim, 
            nhead=num_heads, 
            dim_feedforward=512, 
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            self.decoder_layer, num_layers=num_layers
        )

        # Output Layer
        self.fc_out = nn.Linear(model_dim, output_dim)

    def forward(self, src, tgt):
        # src: (batch_size, seq_len, input_dim), tgt: (batch_size, seq_len, input_dim)
        seq_len = src.size(1)
        src = self.input_embedding(src) + self.positional_encoding[:, :seq_len, :]
        memory = self.transformer_encoder(src)

        tgt = self.input_embedding(tgt) + self.positional_encoding[:, :seq_len, :]
        out = self.transformer_decoder(tgt, memory)

        out = self.fc_out(out)
        
        return out