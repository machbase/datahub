import torch
import torch.nn as nn

class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        RevIN for normalizing and denormalizing data.
        Args:
            num_features (int): Number of features (input dimensions).
            eps (float): A small value for numerical stability.
            affine (bool): Whether to learn scaling and shifting parameters.
            subtract_last (bool): Whether to use the last value for normalization.
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        if self.subtract_last:
            self.last = x[:, -1, :].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x


class BiLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout=0.2, use_revin=False, affine=True, subtract_last=False):
        """
        BiLSTM model with optional RevIN normalization.
        Args:
            input_dim (int): Input feature size (e.g., 1 for univariate time series).
            hidden_dim (int): LSTM hidden state size.
            num_layers (int): Number of LSTM layers.
            output_dim (int): Output dimension (e.g., 1 for single-step forecasting).
            dropout (float): Dropout rate.
            use_revin (bool): Whether to use RevIN normalization.
        """
        super(BiLSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_revin = use_revin

        # RevIN initialization
        if self.use_revin:
            self.revin = RevIN(num_features=input_dim, affine=affine, subtract_last=subtract_last)

        # BiLSTM layer
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                              dropout=dropout, batch_first=True, bidirectional=True)

        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # BiLSTM has hidden_dim * 2

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_dim).
        Returns:
            torch.Tensor: Prediction of shape (batch_size, output_dim).
        """
        # Apply RevIN normalization
        if self.use_revin:
            x = self.revin(x, mode="norm")

        # LSTM output
        out, _ = self.bilstm(x)  # out shape: (batch_size, seq_len, hidden_dim * 2)

        # Use the last time step's output
        out = out[:, -1, :]  # shape: (batch_size, hidden_dim * 2)

        # Fully connected layer
        out = self.fc(out)  # shape: (batch_size, output_dim)

        # Apply RevIN denormalization
        if self.use_revin:
            out = self.revin(out.unsqueeze(1), mode="denorm").squeeze(1)

        return out