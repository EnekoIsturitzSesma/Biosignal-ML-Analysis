import torch.nn as nn
import torch

class ShallowConvNet(nn.Module):
    def __init__(self, num_channels, signal_length, num_classes, K=40):
        super().__init__()

        # Temporal conv
        self.temp_conv = nn.Conv2d(in_channels=1, out_channels=K, kernel_size=(1, 25), bias=True)

        # Spatial conv
        self.spat_conv = nn.Conv2d(in_channels=K, out_channels=K, kernel_size=(num_channels, 1), bias=True)

        self.bn = nn.BatchNorm2d(K, momentum=0.1, eps=1e-5)

        self.avg_pool = nn.AvgPool2d(kernel_size=(1, 75), stride=(1, 15))

        self.dropout = nn.Dropout(p=0.5)

        T_after_temp = signal_length - 25 + 1  
        T_after_pool = (T_after_temp - 75) // 15 + 1  

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(K * T_after_pool, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)         

        x = self.temp_conv(x)      
        x = self.spat_conv(x)       
        x = self.bn(x)

        x = x ** 2                  
        x = self.avg_pool(x)        
        x = torch.log(x.clamp(min=1e-6))   
        x = self.dropout(x)

        x = self.flatten(x)        
        x = self.fc(x)              
        return x

    @torch.no_grad()
    def apply_max_norm(self):
        for layer, c in [(self.temp_conv, 2.0),
                         (self.spat_conv, 2.0),
                         (self.fc,        0.5)]:
            norm = layer.weight.norm(2, dim=0, keepdim=True).clamp(min=1e-8)
            layer.weight.data *= (c / norm).clamp(max=1.0)



class DeepConvNet(nn.Module):
    def __init__(self, num_channels, signal_length, num_classes, dropout_rate=0.5):
        super().__init__()

        self.elu = nn.ELU(alpha=1.0)

        # Block 1
        self.conv1_temp = nn.Conv2d(1,  25, kernel_size=(1, 10), bias=True)
        self.conv1_spat = nn.Conv2d(25, 25, kernel_size=(num_channels, 1), bias=False)
        self.bn1   = nn.BatchNorm2d(25,  momentum=0.1, eps=1e-5)
        self.pool1 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop1 = nn.Dropout(dropout_rate)

        # Block 2
        self.conv2 = nn.Conv2d(25,  50,  kernel_size=(1, 10), bias=False)
        self.bn2   = nn.BatchNorm2d(50,  momentum=0.1, eps=1e-5)
        self.pool2 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop2 = nn.Dropout(dropout_rate)

        # Block 3
        self.conv3 = nn.Conv2d(50,  100, kernel_size=(1, 10), bias=False)
        self.bn3   = nn.BatchNorm2d(100, momentum=0.1, eps=1e-5)
        self.pool3 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))
        self.drop3 = nn.Dropout(dropout_rate)

        # Block 4
        self.conv4 = nn.Conv2d(100, 200, kernel_size=(1, 10), bias=False)
        self.bn4   = nn.BatchNorm2d(200, momentum=0.1, eps=1e-5)
        self.pool4 = nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2))

        # Block 5
        self.flatten = nn.Flatten()
        T_final = self._get_temporal_size(signal_length)
        self.fc = nn.Linear(200 * T_final, num_classes)

    @staticmethod
    def _get_temporal_size(T):
        T = T - 9   # conv1_temp  (conv1_spat no toca la dimensión temporal)
        T = T // 2  # pool1
        T = T - 9   # conv2
        T = T // 2  # pool2
        T = T - 9   # conv3
        T = T // 2  # pool3
        T = T - 9   # conv4
        T = T // 2  # pool4
        return T

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)       

        # Block 1
        x = self.conv1_temp(x)     
        x = self.conv1_spat(x)      
        x = self.bn1(x)
        x = self.elu(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.elu(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.elu(x)
        x = self.pool3(x)
        x = self.drop3(x)

        # Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.elu(x)
        x = self.pool4(x)           

        x = self.flatten(x)
        x = self.fc(x)              
        return x

    @torch.no_grad()
    def apply_max_norm(self):
        for layer in [self.conv1_temp, self.conv1_spat,
                      self.conv2, self.conv3, self.conv4]:
            norm = layer.weight.norm(2, dim=0, keepdim=True).clamp(min=1e-8)
            layer.weight.data *= (2.0 / norm).clamp(max=1.0)

        norm = self.fc.weight.norm(2, dim=0, keepdim=True).clamp(min=1e-8)
        self.fc.weight.data *= (0.5 / norm).clamp(max=1.0)