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


class CNNBiLSTMGait(nn.Module):
    def __init__(self, num_channels, num_class, cnn_channels=64, kernel_size=5, hidden_size=128, num_layers=2, dropout_rate=0.25):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(num_channels, cnn_channels, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels),
            nn.ReLU(),
            nn.Dropout(dropout_rate),

            nn.Conv1d(cnn_channels, cnn_channels*2, kernel_size=kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(cnn_channels*2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        self.lstm = nn.LSTM(input_size=cnn_channels*2, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout_rate, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_class)

    def forward(self, x):
        x = x.permute(0,2,1)   # Shape for Conv1d
        x = self.cnn(x)
        x = x.permute(0,2,1)   # Shape for lstm

        x, _ = self.lstm(x)
        x = self.fc(x)
        return x