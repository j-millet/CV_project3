import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import batched_nms, box_iou

class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.out_channels = 256

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

class Protonet(nn.Module):
    def __init__(self, in_channels, num_prototypes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(256, num_prototypes, kernel_size=1)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.conv2(x)

class PredictionHead(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        return self.conv(x).permute(0, 2, 3, 1).contiguous()

class SimpleInstanceSeg(nn.Module):
    def __init__(self, num_classes, num_prototypes=32):
        super().__init__()
        self.num_classes = num_classes
        self.num_prototypes = num_prototypes
        self.stride = 8
        self.scales = [32]

        self.backbone = Backbone()
        self.protonet = Protonet(256, num_prototypes)
        self.class_head = PredictionHead(256, num_classes + 1)
        self.box_head = PredictionHead(256, 4)
        self.coeff_head = PredictionHead(256, num_prototypes)

    def forward(self, images, targets=None):
        images = torch.stack(images)

        features = self.backbone(images)
        prototypes = self.protonet(features)
        
        anchors = self.generate_anchors(features, images.shape[2:])
        
        class_preds = self.class_head(features).view(features.size(0), -1, self.num_classes + 1)
        box_preds = self.box_head(features).view(features.size(0), -1, 4)
        coeff_preds = self.coeff_head(features).view(features.size(0), -1, self.num_prototypes)

        if targets is not None:
            return self.compute_loss(anchors, class_preds, box_preds, coeff_preds, prototypes, targets)
        else:
            return self.predict(anchors, class_preds, box_preds, coeff_preds, prototypes)

    def generate_anchors(self, features, image_size):
        N, _, H, W = features.shape
        anchors = []
        for i in range(H):
            for j in range(W):
                x_stride = image_size[1] / W  # 512 / W
                y_stride = image_size[0] / H  # 256 / H
                cx = (j + 0.5) * x_stride
                cy = (i + 0.5) * y_stride
                for scale in self.scales:
                    anchors.append([cx - scale/2, cy - scale/2, cx + scale/2, cy + scale/2])
        return torch.tensor(anchors, device=features.device).repeat(N, 1, 1)

    def compute_loss(self, anchors, class_preds, box_preds, coeff_preds, prototypes, targets):
        device = anchors.device
        cls_loss = torch.tensor(0., device=device, requires_grad=True)
        box_loss = torch.tensor(0., device=device, requires_grad=True)
        mask_loss = torch.tensor(0., device=device, requires_grad=True)
        
        valid_samples = 0
        
        for i in range(len(targets)):
            gt_boxes = targets[i]['boxes']
            if gt_boxes.shape[0] == 0:
                continue
                
            ious = box_iou(anchors[i], gt_boxes)
            max_ious, gt_ids = ious.max(1)
            pos_mask = max_ious > 0.5
            
            if not pos_mask.any():
                cls_loss = cls_loss + 0.*class_preds[i].mean()
                box_loss = box_loss + 0.*box_preds[i].mean()
                continue
                
            valid_samples += 1
            
            cls_loss = cls_loss + F.cross_entropy(
                class_preds[i][pos_mask], 
                targets[i]['labels'][gt_ids[pos_mask]]
            )
            
            pos_anchors = anchors[i][pos_mask]
            gt_boxes_matched = gt_boxes[gt_ids[pos_mask]]
            box_targets = self.box_to_delta(pos_anchors, gt_boxes_matched)
            box_loss = box_loss + F.smooth_l1_loss(
                box_preds[i][pos_mask], 
                box_targets
            )
            
            matched_instance_ids = gt_ids[pos_mask]
            coeffs = coeff_preds[i][pos_mask]
            
            for coeff, instance_id in zip(coeffs, matched_instance_ids):
                proto = prototypes[i]
                gt_mask = targets[i]['masks'][instance_id].float()
                
                gt_mask = F.interpolate(
                    gt_mask.unsqueeze(0).unsqueeze(0),
                    size=proto.shape[1:],
                    mode='nearest'
                ).squeeze()
                
                mask_pred = (proto * coeff.view(-1, 1, 1)).sum(0).sigmoid()
                mask_pred = torch.clamp(mask_pred, 1e-6, 1-1e-6)
                
                mask_loss = mask_loss + F.binary_cross_entropy(
                    mask_pred, 
                    (gt_mask > 0.5).float()
                )

        valid_samples = max(valid_samples, 1)
        
        return {
            'loss_classifier': cls_loss / valid_samples,
            'loss_box_reg': box_loss / valid_samples,
            'loss_mask': mask_loss / valid_samples if mask_loss > 0 else cls_loss * 0
        }

    def box_to_delta(self, anchors, gt_boxes):
        a_cxcy = (anchors[:, :2] + anchors[:, 2:]) / 2
        a_wh = anchors[:, 2:] - anchors[:, :2]
        gt_cxcy = (gt_boxes[:, :2] + gt_boxes[:, 2:]) / 2
        gt_wh = gt_boxes[:, 2:] - gt_boxes[:, :2]
        return torch.cat([
            (gt_cxcy - a_cxcy) / a_wh,
            torch.log(gt_wh / a_wh)
        ], dim=1)

    def predict(self, anchors, class_preds, box_preds, coeff_preds, prototypes):
        class_probs = F.softmax(class_preds.view(len(anchors), -1, self.num_classes + 1), -1)[..., :-1]
        scores, labels = class_probs.max(-1)
        
        all_preds = []
        for i in range(len(anchors)):
            boxes = self.delta_to_boxes(anchors[i], box_preds[i].view(-1, 4))
            keep = batched_nms(boxes, scores[i], labels[i], 0.5)[:100]
            
            pred_masks = []
            for coeff in coeff_preds[i].view(-1, self.num_prototypes)[keep]:
                raw_mask = (prototypes[i] * coeff.view(-1, 1, 1)).sum(0).sigmoid()
                final_mask = F.interpolate(
                    raw_mask.unsqueeze(0).unsqueeze(0),
                    size=(256, 512),
                    mode='bilinear',
                    align_corners=False
                ).squeeze()
                pred_masks.append(final_mask > 0.5)
            
            all_preds.append({
                'boxes': boxes[keep],
                'labels': labels[i][keep],
                'scores': scores[i][keep],
                'masks': torch.stack(pred_masks) if pred_masks else torch.tensor([])
            })
        return all_preds

    def delta_to_boxes(self, anchors, deltas):
        a_cxcy = (anchors[:, :2] + anchors[:, 2:]) / 2
        a_wh = anchors[:, 2:] - anchors[:, :2]
        pred_cxcy = a_cxcy + deltas[:, :2] * a_wh
        pred_wh = a_wh * torch.exp(deltas[:, 2:])
        return torch.cat([
            pred_cxcy - pred_wh/2,
            pred_cxcy + pred_wh/2
        ], dim=1)