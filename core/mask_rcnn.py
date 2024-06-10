'''
    Mask Region-based Convolutional Neural Network (Mask RCNN) class
'''
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, TensorDataset

from .data_handler import BinaryHandler
from .convolutional import VoxelBackbone, RegionProposalNetwork, ROIAlign3D, RCNNHead

class MaskRCNN(nn.Module):
    def __init__(self, num_classes):
        super(MaskRCNN, self).__init__()
        self.backbone = VoxelBackbone()
        self.rpn = RegionProposalNetwork(in_channels=256, mid_channels=256, num_anchors=9)
        self.roi_align = ROIAlign3D(output_size=(7, 7, 7))
        self.rcnn_head = RCNNHead(input_dim=256 * 7 * 7 * 7, num_classes=num_classes)
        
    def forward(self, x, proposals):
        feature_map = self.backbone(x)
        rpn_cls_logits, rpn_bbox_pred = self.rpn(feature_map)
        pooled_features = self.roi_align(feature_map, proposals)
        
        num_proposals = pooled_features.size(0)
        input_dim = pooled_features.size(1) * pooled_features.size(2) * pooled_features.size(3) * pooled_features.size(4)
        flattened_features = pooled_features.view(num_proposals, -1)
        
        cls_score, bbox_pred, mask_logits = self.rcnn_head(flattened_features, pooled_features)
        
        return cls_score, bbox_pred, mask_logits
