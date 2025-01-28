import torchvision.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign
from torchvision.ops import generalized_box_iou_loss

def compute_losses(pred_masks, pred_boxes, targets):
    """
    Computes mask and bounding box losses for instance segmentation.
    
    Args:
        pred_masks (torch.Tensor): Predicted masks (B, N, H, W)
        pred_boxes (torch.Tensor): Predicted bounding boxes (B, N, 4)
        targets (list[dict]): Ground truth targets with keys:
            - 'masks': Ground truth masks (B, N, H, W)
            - 'boxes': Ground truth boxes (B, N, 4)
        mask_threshold (float): Threshold for binary mask classification.
        
    Returns:
        dict: Loss dictionary with `loss_mask` and `loss_box`.
    """
    loss_dict = {}
    
    # Flatten predictions and targets
    target_masks = torch.cat([t['masks'] for t in targets], dim=0).float()  # (T, H, W)
    target_boxes = torch.cat([t['boxes'] for t in targets], dim=0)          # (T, 4)
    pred_masks = pred_masks.sigmoid()                                       # Apply sigmoid activation
    pred_boxes = torch.cat(pred_boxes, dim=0)                               # (T, 4)

    # --- Mask Loss ---
    # Resize target masks to match predicted mask size
    pred_masks_resized = F.interpolate(pred_masks, size=target_masks.shape[-2:], mode="bilinear", align_corners=False)
    
    # Binary Cross-Entropy for each pixel
    loss_mask = F.binary_cross_entropy(pred_masks_resized, target_masks, reduction="mean")
    loss_dict["loss_mask"] = loss_mask

    # --- Bounding Box Loss ---
    # Use Generalized IoU Loss for bounding boxes
    loss_box = generalized_box_iou_loss(pred_boxes, target_boxes, reduction="mean")
    loss_dict["loss_box"] = loss_box
    
    return loss_dict

class Custom1(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Custom Backbone
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),  # Input -> 64 channels
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._conv_block(64, 128),
            self._conv_block(128, 256),
            self._conv_block(256, 512),
        )
        
        # Feature Pyramid
        self.fpn = nn.ModuleList([
            nn.Conv2d(512, 256, kernel_size=1),
            nn.Conv2d(512, 256, kernel_size=3, padding=1)
        ])
        
        # RoIAlign for instance features
        self.roi_align = MultiScaleRoIAlign(
            featmap_names=["0", "1"],
            output_size=(7, 7),
            sampling_ratio=2
        )
        
        # Mask Prediction Head
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, num_classes, kernel_size=1)
        )
    
    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
    
    def forward(self, images, targets):
        # Extract image shapes before stacking
        image_shapes = [img.shape[-2:] for img in images]  # (Height, Width) for each image

        # Stack the list of images into a single batch tensor
        images = torch.stack(images, dim=0)

        # Backbone feature extraction
        features = self.backbone(images)  # Single Tensor from the backbone

        # FPN feature maps (list of tensors)
        fpn_features = [fpn_layer(features) for fpn_layer in self.fpn]  # List of tensors

        # Concatenate FPN feature maps along the channel dimension
        fpn_features = [F.interpolate(f, size=fpn_features[0].shape[-2:], mode="bilinear", align_corners=False)
                        for f in fpn_features]
        fpn_features = torch.cat(fpn_features, dim=1)  # Concatenate along the channel dimension

        # RoIAlign for region proposals (assume we have targets with "boxes")
        rois = self.roi_align(
            fpn_features,  # Feature maps
            [t["boxes"] for t in targets],  # List of bounding boxes
            image_shapes  # Original image sizes
        )

        # Mask Prediction
        mask_logits = self.mask_head(rois)

        # Loss Calculation
        losses = compute_losses(mask_logits, rois, targets)
        return losses

class Custom2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        # Use ResNet only for the first two layers
        resnet = models.resnet34(weights=None)
        self.resnet_base = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2
        )
        
        # Custom layers from here
        self.conv_block = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
        )
        
        # Mask Prediction Head
        self.mask_head = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )
    
    def forward(self, images, targets):
        # ResNet Feature Extraction
        features = self.resnet_base(images)
        
        # Additional Convolutions
        features = self.conv_block(features)
        
        # Mask Prediction
        mask_logits = self.mask_head(features)
        
        # Compute losses
        losses = compute_losses(mask_logits, features, targets)
        # Add loss calculations here
        return losses
