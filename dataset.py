import os
import torch
from torchvision import tv_tensors
from torchvision.transforms.v2 import functional as F
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import functional
from PIL import Image


class CityscapesSimplifiedDataset(torch.utils.data.Dataset):
    def __init__(self, root, split, transforms, target_size):
        self.image_dir = os.path.join(root, "leftImg8bit", split)
        self.target_dir = os.path.join(root, "gtFine", split)
        self.transforms = transforms
        self.target_size = target_size

        self.images = []
        self.targets = []

        for city in os.listdir(self.image_dir):
            img_dir = os.path.join(self.image_dir, city)
            target_dir = os.path.join(self.target_dir, city)
            for file_name in os.listdir(img_dir):
                target_name = "{}_{}".format(
                    file_name.split("_leftImg8bit")[0], "gtFine_instanceIds.png"
                )
                target = os.path.join(target_dir, target_name)
                self.images.append(os.path.join(img_dir, file_name))
                self.targets.append(target)
        
        self.images.sort()
        self.targets.sort()

    def __getitem__(self, idx):
        image_path = self.images[idx]
        annotation_path = self.targets[idx]

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(annotation_path)

        image = F.resize(image, self.target_size)
        mask = F.resize(mask, self.target_size, interpolation=F.InterpolationMode.NEAREST)

        image = functional.pil_to_tensor(image).to(dtype=torch.float32) / 255.0
        mask = functional.pil_to_tensor(mask).to(dtype=torch.int64)
        
        mask[mask < 34] = 0 # only extract classes that have instances

        obj_ids = torch.unique(mask)
        obj_ids = obj_ids[1:]  # ignore background

        # create binary masks for each object
        masks = (mask == obj_ids[:, None, None]).to(dtype=torch.uint8)
        valid_masks = masks.sum(dim=(1, 2)) > 0
        masks = masks[valid_masks]
        obj_ids = obj_ids[valid_masks]

        # get labels
        labels = (obj_ids // 1000).to(dtype=torch.int64)
        labels -= 23  # Shift to 0-based indexing

        # bounding boxes
        boxes = masks_to_boxes(masks)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        valid_boxes = area > 0
        boxes = boxes[valid_boxes]
        masks = masks[valid_boxes]
        obj_ids = obj_ids[valid_boxes]
        area = area[valid_boxes]

        # iscrowd
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        image = tv_tensors.Image(image)
        target = {
            "boxes": tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=F.get_size(image)),
            "masks": tv_tensors.Mask(masks),
            "labels": labels,
            "image_id": idx,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self):
        return len(self.images)


class InferenceDataset(torch.utils.data.Dataset):
    def __init__(self, images:list , transforms, target_size):
        super(InferenceDataset).__init__()
        
        self.images = images
        self.transforms = transforms
        self.target_size = target_size
        
    def __getitem__(self, idx):
        image = self.images[idx]
        image = F.resize(image, self.target_size)
        
        image = functional.pil_to_tensor(image).to(dtype=torch.float32) / 255.0
        
        image = tv_tensors.Image(image)
        
        if self.transforms is not None:
            image = self.transforms(image)
        
        return image, None # No target for inference
    
    def __len__(self):
        return len(self.images)