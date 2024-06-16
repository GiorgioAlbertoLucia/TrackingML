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
    def __init__(self):
        super(MaskRCNN, self).__init__()
        self.backbone = VoxelBackbone()
        self.rpn = RegionProposalNetwork(in_channels=256, mid_channels=256, num_anchors=9)
        self.roi_align = ROIAlign3D(output_size=(7, 7, 7))
        self.rcnn_head = RCNNHead(input_dim=256 * 7 * 7 * 7, num_classes=2) # 2 classes: background and signal
        
    def forward(self, x):
        
        feature_map = self.backbone(x)
        proposals, _ = self.rpn(feature_map)
        
        pooled_features = self.roi_align(feature_map, proposals)
        class_scores, bbox_preds, mask_logits = self.rcnn_head(pooled_features.view(pooled_features.size(0), -1), pooled_features)

        mask_probs = torch.sigmoid(mask_logits)
        mask_probs = mask_probs.detach().cpu().numpy()

        mask = np.zeros_like(x)
        for proposal, prob in zip(proposals, mask_probs):
            x1, z1, y1, x2, z2, y2 = map(int, proposal)
            pred_class = np.argmax(prob)
            mask[0, 0, z1:z2, y1:y2, x1:x2] = pred_class

        return mask
