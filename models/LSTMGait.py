import torch.nn as nn

class LSTMGait(nn.Module):
    def __init__(self, num_channels, num_class, hidden_size=128, num_layers=2, dropout_rate=0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_size=num_channels, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate)
        self.fc = nn.Linear(hidden_size, num_class)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x)
        return x