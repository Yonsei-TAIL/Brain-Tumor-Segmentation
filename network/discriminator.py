import torch.nn as nn
import torch
import torch.nn.functional as F

class MaskDiscriminator(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(MaskDiscriminator, self).__init__()

        self.conv1 = self.conv_block(in_channels + n_classes, 64)
        self.conv2 = self.conv_block(64, 128)
        self.conv3 = self.conv_block(128, 256)
        self.conv4 = self.conv_block(256, 512)
        self.conv5 = self.conv_block(512, 512)

        self.gap = nn.AdaptiveAvgPool2d((1,1))

        self.fc1 = nn.Linear(512, 256, bias=False)
        self.fc2 = nn.Linear(256, 1, bias=False)

    def conv_block(self, feat_in, feat_out):
        return nn.Sequential(
            nn.Conv2d(feat_in, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_out),
            nn.ReLU(),
            nn.Conv2d(feat_out, feat_out, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(feat_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, img, mask):
        x = torch.cat([img, mask], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x