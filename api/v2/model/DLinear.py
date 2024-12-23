import torch
import torch.nn as nn

## modeling
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
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
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x
    
class moving_avg(torch.nn.Module):
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = torch.nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class series_decomp(torch.nn.Module):
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        residual = x - 0.3*moving_mean
        return moving_mean, residual 
        
class DLinear(torch.nn.Module):
    def __init__(self, window_size, forecast_size, kernel_size, individual, feature_size, use_revin=False, multi_feature=False, affine=True, subtract_last=False):
        """
        :param window_size: input sequence length
        :param forecast_size: number of timesteps to forecast
        :param kernel_size: kernel size for decomposition
        :param individual: whether to use individual linear layers for each feature
        :param feature_size: number of features (channels)
        :param use_revin: whether to apply RevIN for normalization
        :param multi_feature: whether to apply FC layer for multi feature 
        """
        super(DLinear, self).__init__()
        self.window_size = window_size
        self.forecast_size = forecast_size
        self.decomposition = series_decomp(kernel_size)
        self.individual = individual
        self.channels = feature_size
        self.use_revin = use_revin
        self.multi_feature = multi_feature

        # Initialize RevIN
        if self.use_revin:
            self.revin = RevIN(num_features=self.channels, affine=affine, subtract_last=subtract_last)

        # Linear layers
        if self.individual:
            self.Linear_Trend = torch.nn.ModuleList()
            self.Linear_Seasonal = torch.nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Trend.append(torch.nn.Linear(self.window_size, self.forecast_size))
                self.Linear_Trend[i].weight = torch.nn.Parameter((1 / self.window_size) * torch.ones([self.forecast_size, self.window_size]))
                self.Linear_Seasonal.append(torch.nn.Linear(self.window_size, self.forecast_size))
                self.Linear_Seasonal[i].weight = torch.nn.Parameter((1 / self.window_size) * torch.ones([self.forecast_size, self.window_size]))
        else:
            self.Linear_Trend = torch.nn.Linear(self.window_size, self.forecast_size)
            self.Linear_Trend.weight = torch.nn.Parameter((1 / self.window_size) * torch.ones([self.forecast_size, self.window_size]))
            self.Linear_Seasonal = torch.nn.Linear(self.window_size, self.forecast_size)
            self.Linear_Seasonal.weight = torch.nn.Parameter((1 / self.window_size) * torch.ones([self.forecast_size, self.window_size]))
            
        if self.multi_feature:
            self.fc = torch.nn.Linear(feature_size , 1)
                

    def forward(self, x):
        """
        :param x: input tensor of shape (batch_size, seq_len, num_features)
        :return: output tensor of shape (batch_size, forecast_size, num_features)
        """
        # Normalize input using RevIN
        if self.use_revin:
            x = self.revin(x, mode="norm")

        # Decompose input into trend and seasonal components
        trend_init, seasonal_init = self.decomposition(x)
        trend_init, seasonal_init = trend_init.permute(0, 2, 1), seasonal_init.permute(0, 2, 1)

        # Apply linear layers
        if self.individual:
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.forecast_size], dtype=trend_init.dtype).to(trend_init.device)
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.forecast_size], dtype=seasonal_init.dtype).to(seasonal_init.device)
            for idx in range(self.channels):
                trend_output[:, idx, :] = self.Linear_Trend[idx](trend_init[:, idx, :])
                seasonal_output[:, idx, :] = self.Linear_Seasonal[idx](seasonal_init[:, idx, :])
        else:
            trend_output = self.Linear_Trend(trend_init)
            seasonal_output = self.Linear_Seasonal(seasonal_init)

        # Combine components
        x = seasonal_output + trend_output

        # Denormalize output using RevIN
        if self.use_revin:
            x = self.revin(x.permute(0, 2, 1), mode="denorm").permute(0, 2, 1)
            
        if self.multi_feature:
            x = self.fc(x.squeeze(2))    

        return x