import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math

class CDFIM(nn.Module):
    """Channel Difference Feature Interaction Module"""
    
    def __init__(self, dim, kernel_size=3, dilation=1, reduction=16):
        super().__init__()
        
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=1, bias=True)
        self.act = nn.GELU()
        
        # Channel attention
        self.ca = CPCA_ChannelAttention(dim, reduction)
        
        # Depthwise separable convolutions
        self.conv3_1 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.conv3_2 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=2, dilation=2, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.conv3_3 = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=3, dilation=3, groups=dim, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )
        
        self.conv1_1 = nn.Conv2d(dim*3, dim, kernel_size=1, bias=True)
        
    def forward(self, x):
        # x: list of [rgb_features, ir_features]
        inputs = x[0] - x[1]  # channel difference
        
        inputs = self.conv1(inputs)
        inputs = self.act(inputs)
        inputs = self.ca(inputs)
        
        # Feature interaction
        inputs0 = inputs + x[0]
        inputs1 = inputs + x[1]
        inputs = inputs0 + inputs1
        
        # Multi-scale feature extraction
        out1 = self.conv3_1(inputs)
        out2 = self.conv3_2(inputs)
        out3 = self.conv3_3(inputs)
        
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv1_1(out)
        
        return out

class CGSA(nn.Module):
    """Cross-Gated Spatial Attention"""
    
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        
        self.proj1 = Freprocess(dim, kernel_size)
        self.proj2 = Freprocess(dim, kernel_size)
        
        # Channel attention
        self.ch_wv = nn.Conv2d(dim, dim//2, kernel_size=1)
        self.ch_wq = nn.Conv2d(dim, dim//2, kernel_size=1)
        
        # Gated multimodal layer
        self.ga1 = GatedMultimodalLayer(dim, dim, dim)
        self.ga2 = GatedMultimodalLayer(dim, dim, dim)
        
        self.conv = nn.Conv2d(dim*2, dim, kernel_size=1, bias=True)
        
    def forward(self, rgb, ir):
        b, c, h, w = rgb.shape
        
        # Preprocess
        rgb = self.proj1(rgb)
        ir = self.proj2(ir)
        
        # Channel attention
        channel_wv_rgb = self.ch_wv(rgb)
        channel_wq_rgb = self.ch_wq(rgb)
        
        channel_wv_rgb = channel_wv_rgb.reshape(b, c//2, -1)
        channel_wq_rgb = channel_wq_rgb.reshape(b, -1, 1)
        
        channel_att_rgb = torch.matmul(channel_wv_rgb, channel_wq_rgb)
        channel_att_rgb = channel_att_rgb.reshape(b, 1, h, w)
        channel_att_rgb = torch.sigmoid(channel_att_rgb)
        
        out_rgb = rgb * channel_att_rgb
        
        # Cross-modal interaction
        channel_out_rgb = self.ga1(ir, rgb)
        channel_out_ir = self.ga2(out_rgb, ir)
        
        out = torch.cat((channel_out_rgb, channel_out_ir), 1)
        out = self.conv(out)
        
        return out

# Supporting classes
class CPCA_ChannelAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class GatedMultimodalLayer(nn.Module):
    def __init__(self, size_in1, size_in2, size_out):
        super().__init__()
        self.fc1 = nn.Linear(size_in1, size_out)
        self.fc2 = nn.Linear(size_in2, size_out)
        self.fc3 = nn.Linear(size_out*2, size_out)
        
    def forward(self, x1, x2):
        # Simple concatenation for now
        return torch.cat([x1, x2], dim=1)

class Freprocess(nn.Module):
    def __init__(self, dim, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, kernel_size, padding=kernel_size//2, groups=dim)
        
    def forward(self, x):
        return self.conv(x)