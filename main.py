import sys

import torch
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms.v2 as T

import utils
from engine import train_one_epoch, evaluate, train_own_model
from dataset import CityscapesSimplifiedDataset
from models import  Custom1, Custom2
from optimizers import get_optimizer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
TARGET_SIZE = (256, 512)
THRESHOLD = 0.7
MODEL_PATH = ''

camera_calibration_paths = {
    'train': "data/camera/train",
    'test': "data/camera/test",
    'val': "data/camera/val",
}
image_paths = {
    'train': "data/leftImg8bit/train",
    'test': "data/leftImg8bit/test",
    'val': "data/leftImg8bit/val",
}
annotation_paths = {
    'train': "data/gtFine/train",
    'test': "data/gtFine/test",
    'val': "data/gtFine/val",
}

def get_transform(train):
    transforms = []
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    transforms.append(T.ToDtype(torch.float, scale=True))
    transforms.append(T.ToPureTensor())
    return T.Compose(transforms)

datasets = {
    split: CityscapesSimplifiedDataset(
        root="data",
        split=split,
        transforms=get_transform(train=(split == "train")),
        target_size=TARGET_SIZE,
    )
    for split in ["train", "test", "val"]
}

dataloaders = {
    split: DataLoader(
        datasets[split],
        batch_size=4,
        shuffle=(split == "train"),
        num_workers=4,
        collate_fn=utils.collate_fn,
        pin_memory=True
    )
    for split in ["train", "test", "val"]
}

def setup_model(path=None):
    model = maskrcnn_resnet50_fpn()
    num_classes = 11
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    
    if path: model.load_state_dict(torch.load(path, weights_only=True))
    model.to(DEVICE)
    return model

def train(model, num_epochs=20):
    writer = SummaryWriter(log_dir="data/runs/cityscapes_mask_rcnn")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params,
        lr=0.005,
        momentum=0.9,
        weight_decay=0.0005
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, dataloaders['train'], DEVICE, THRESHOLD, epoch, writer, print_freq=10)
        lr_scheduler.step()
        evaluate(model, dataloaders['val'], DEVICE)
        torch.save(model.state_dict(), 'data/model_checkpoints/mask_rcnn_model.pth')

if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    model_num = input('Choose model\n1. Mask R-CNN\n2. Custom1\n3. Custom2\nChoice: ')
    checkpoint = input('Do you want to load checkpoint? (y/n): ')
    if model_num == '1':
        MODEL_PATH = 'data/model_checkpoints/mask_rcnn_model.pth' if checkpoint == 'y' else None
        model = setup_model(MODEL_PATH)
        train(model)

    elif model_num == '2':
        MODEL_PATH = 'data/model_checkpoints/custom1_model.pth' if checkpoint == 'y' else None
        model = Custom1(num_classes=11).to(DEVICE)
        optimizer_names = ['adamw', 'rmsprop', 'nesterov']
        for optimizer_name in optimizer_names:
            optimizer = get_optimizer(model, optimizer_name)
            writer = SummaryWriter(log_dir=f"data/runs/cityscapes_custom1_{optimizer_name}")
            train_own_model('custom1', model, optimizer, dataloaders["train"], dataloaders['val'], DEVICE, THRESHOLD, 10, writer)

    elif model_num == '3':
        MODEL_PATH = 'data/model_checkpoints/custom2_model.pth' if checkpoint == 'y' else None
        model = Custom2(num_classes=11).to(DEVICE)
        optimizer_names = ['adamw', 'rmsprop', 'nesterov']
        for optimizer_name in optimizer_names:
            optimizer = get_optimizer(model, optimizer_name)
            writer = SummaryWriter(log_dir=f"data/runs/cityscapes_custom2_{optimizer_name}")
            train_own_model('custom2', model, optimizer, dataloaders["train"], dataloaders['val'], DEVICE, THRESHOLD, 10, writer)

    else:
        print('Invalid choice')
        sys.exit(1)
    
    utils.show_results(model, dataloaders["test"], DEVICE, THRESHOLD)