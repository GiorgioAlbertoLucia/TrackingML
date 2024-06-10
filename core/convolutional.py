'''
    Convolutional Neural Network classes
'''
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from .data_handler import BinaryHandler

class VoxelBackbone(nn.Module):
    '''
        Convolutional neural network used to extract the feature maps out of the voxel grid
    '''
    
    def __init__(self):
        super(VoxelBackbone, self).__init__()
        # Define the convolutional layers
        self.conv1 = nn.Conv3d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, stride=1, padding=1)
        # Define the pooling layer
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        # Apply the first convolution and pooling layers
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        # Apply the second convolution and pooling layers
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        # Apply the third convolution and pooling layers
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        return x
    
class RegionProposalNetwork(nn.Module):
    '''
        Neural network used to propose regions of interest in the feature maps
    '''

    def __init__(self, in_channels, mid_channels=256, num_anchors=9):
        super(RegionProposalNetwork, self).__init__()
        self.conv = nn.Conv3d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1)
        self.cls_logits = nn.Conv3d(mid_channels, num_anchors * 2, kernel_size=1, stride=1)
        self.bbox_pred = nn.Conv3d(mid_channels, num_anchors * 6, kernel_size=1, stride=1)

    def forward(self, x):
        x = F.relu(self.conv(x))
        rpn_cls_logits = self.cls_logits(x)
        rpn_bbox_pred = self.bbox_pred(x)
        return rpn_cls_logits, rpn_bbox_pred

class ROIAlign3D(nn.Module):
    '''
        Region of Interest Align layer for 3D feature maps
    '''

    def __init__(self, output_size):
        super(ROIAlign3D, self).__init__()
        self.output_size = output_size

    def forward(self, feature_map, proposals):
        pooled_features = []
        for proposal in proposals:
            batch_index, z1, y1, x1, z2, y2, x2 = proposal
            z1, y1, x1, z2, y2, x2 = int(z1), int(y1), int(x1), int(z2), int(y2), int(x2)
            region = feature_map[batch_index, :, z1:z2, y1:y2, x1:x2]
            pooled_region = nn.functional.adaptive_max_pool3d(region, self.output_size)
            pooled_features.append(pooled_region)
        pooled_features = torch.stack(pooled_features)
        return pooled_features
    
class MaskHead(nn.Module):
    '''
        Neural network used to predict the mask  of the object in the region of interest
    '''
    def __init__(self, input_channels, num_classes):
        super(MaskHead, self).__init__()
        self.conv1 = nn.Conv3d(input_channels, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv3d(256, 256, kernel_size=3, stride=1, padding=1)
        self.deconv = nn.ConvTranspose3d(256, 256, kernel_size=2, stride=2)
        self.mask_pred = nn.Conv3d(256, num_classes, kernel_size=1)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.deconv(x))
        mask_logits = self.mask_pred(x)
        return mask_logits
    
class RCNNHead(nn.Module):
    '''
        Neural network used to predict the class, bounding box and mask of the object
    '''
    def __init__(self, input_dim, num_classes):
        super(RCNNHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 6)
        self.mask_head = MaskHead(256, num_classes)  # Assuming pooled features have 256 channels
        
    def forward(self, x, mask_features):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)
        mask_logits = self.mask_head(mask_features)
        return cls_score, bbox_pred, mask_logits