import torch.nn as nn

class EEGNet(nn.Module): 
    def __init__(self, num_channels, signal_length, num_class, f1=8, D=2, dropout_rate=0.25):
        super().__init__()
        f2 = f1*D

        self.Elu = nn.ELU()

        #layer 1
        self.TemporalConv = nn.Conv2d(1, f1, (1, 64), padding=(0, 32), bias=False)
        self.BatchNorm1 = nn.BatchNorm2d(f1)

        # layer 2
        self.DepthSpatialConv = nn.Conv2d(f1, f2, (num_channels, 1), groups= f1)
        self.BatchNorm2 = nn.BatchNorm2d(f2)
        self.AvgPool1 = nn.AvgPool2d((1, 4))
        self.Dropout = nn.Dropout2d(dropout_rate)

        # layer 3
        self.SeparableConvDepth = nn.Conv2d( f2, f2, (1, 16), padding=(0, 8), groups= f2)
        self.SeparableConvPoint = nn.Conv2d(f2, f2, (1, 1))
        self.BatchNorm3 = nn.BatchNorm2d(f2)
        self.AvgPool2 = nn.AvgPool2d((1, 8))

        # layer 4
        self.Flatten = nn.Flatten()
        self.FC = nn.Linear(f2*(signal_length//32), num_class)
        
        
    def forward(self, x):
        x = x.unsqueeze(1)

        # layer 1
        y = self.TemporalConv(x)
        y = self.BatchNorm1(y)
        y = self.Elu(y)
        y = self.Dropout(y)

        # layer 2
        y = self.DepthSpatialConv(y)
        y = self.BatchNorm2(y)
        y = self.Elu(y)
        y = self.AvgPool1(y)
        y = self.Dropout(y)

        # layer 3
        y = self.SeparableConvDepth(y)
        y = self.SeparableConvPoint(y)
        y = self.BatchNorm3(y)
        y = self.Elu(y)
        y = self.AvgPool2(y)
        y = self.Dropout(y)

        # layer 4
        y = self.Flatten(y)
        y = self.FC(y)
        
        return y