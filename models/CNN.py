import torch.nn as nn

class ShallowConvNet(nn.Module):
    def __init__(self, num_channels, signal_length, num_class, num_filters=16, hidden_units=64, dropout_rate=0.25):
        super().__init__()

        self.activation = nn.Sigmoid()

        self.Conv = nn.Conv2d(in_channels=1, out_channels=num_filters, kernel_size=(num_channels, 5), padding=(0, 2), bias=False)

        self.BatchNorm = nn.BatchNorm2d(num_filters)

        self.AvgPool = nn.AvgPool2d((1, 4))

        self.Dropout = nn.Dropout(dropout_rate)

        self.Flatten = nn.Flatten()

        pooled_length = signal_length // 4
        self.feature_size = num_filters * pooled_length

        self.FC1 = nn.Linear(self.feature_size, hidden_units)
        self.FC2 = nn.Linear(hidden_units, num_class)

    def forward(self, x):

        x = x.unsqueeze(1)  

        # Conv layer
        y = self.Conv(x)
        y = self.BatchNorm(y)
        y = self.activation(y)

        # Pooling
        y = self.AvgPool(y)
        y = self.Dropout(y)

        # Flatten
        y = self.Flatten(y)

        # MLP
        y = self.FC1(y)
        y = self.activation(y)

        y = self.FC2(y)

        return y



class DeepConvNet(nn.Module):

    def __init__(self, num_channels, signal_length, num_class, dropout_rate=0.5):
        super().__init__()

        self.elu = nn.ELU()

        # Block 1 
        self.temporal_conv = nn.Conv2d(1, 25, (1, 10), padding=(0,5), bias=False)

        self.spatial_conv = nn.Conv2d(25, 25, (num_channels, 1), bias=False)

        self.bn1 = nn.BatchNorm2d(25)

        self.pool1 = nn.MaxPool2d((1, 3))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2 
        self.conv2 = nn.Conv2d(25, 50, (1,10), padding=(0,5), bias=False)
        self.bn2 = nn.BatchNorm2d(50)

        self.pool2 = nn.MaxPool2d((1,3))
        self.drop2 = nn.Dropout(dropout_rate)

        # Block 3 
        self.conv3 = nn.Conv2d(50, 100, (1,10), padding=(0,5), bias=False)
        self.bn3 = nn.BatchNorm2d(100)

        self.pool3 = nn.MaxPool2d((1,3))
        self.drop3 = nn.Dropout(dropout_rate)

        # Block 4 
        self.conv4 = nn.Conv2d(100, 200, (1,10), padding=(0,5), bias=False)
        self.bn4 = nn.BatchNorm2d(200)

        self.pool4 = nn.MaxPool2d((1,3))
        self.drop4 = nn.Dropout(dropout_rate)

        # Classifier 
        self.flatten = nn.Flatten()

        final_time = signal_length // (3*3*3*3)

        self.fc = nn.Linear(200 * final_time, num_class)

    def forward(self, x):

        x = x.unsqueeze(1)  

        # Block 1
        y = self.temporal_conv(x)
        y = self.spatial_conv(y)
        y = self.bn1(y)
        y = self.elu(y)
        y = self.pool1(y)
        y = self.drop1(y)

        # Block 2
        y = self.conv2(y)
        y = self.bn2(y)
        y = self.elu(y)
        y = self.pool2(y)
        y = self.drop2(y)

        # Block 3
        y = self.conv3(y)
        y = self.bn3(y)
        y = self.elu(y)
        y = self.pool3(y)
        y = self.drop3(y)

        # Block 4
        y = self.conv4(y)
        y = self.bn4(y)
        y = self.elu(y)
        y = self.pool4(y)
        y = self.drop4(y)

        # Classifier
        y = self.flatten(y)
        y = self.fc(y)

        return y