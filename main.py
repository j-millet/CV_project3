import sys

import torch
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

import utils
from engine import train_one_epoch, evaluate
from dataset import CityscapesSimplifiedDataset
from models import SimpleInstanceSeg
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


def train(model, model_name, optimizer, num_epochs=10):
    writer = SummaryWriter(log_dir=f"data/runs/cityscapes_{model_name}")
    
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=3,
        gamma=0.1
    )
    
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, dataloaders['train'], DEVICE, THRESHOLD, epoch, writer, print_freq=10)
        lr_scheduler.step()
        evaluate(model, dataloaders['val'], DEVICE)
        torch.save(model.state_dict(), f'data/model_checkpoints/{model_name}_model.pth')


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter

    model_names = ['mask_rcnn', 'InstSegNet']
    optimizer_names = ['adamw', 'rmsprop', 'sgd']

    model_num = input('Choose model\n1. Mask R-CNN\n2. InstSegNet\nChoice: ')
    checkpoint = input('Do you want to load a checkpoint? (y/n): ')

    if model_num in ['1', '2']: 
        MODEL_PATH = f'data/model_checkpoints/{model_names[int(model_num) - 1]}_model.pth' if checkpoint == 'y' else None

    if model_num == '1':
        model = utils.setup_rcnn(MODEL_PATH,DEVICE)
        optimizer = get_optimizer(model, 'sgd')
        train(model, model_names[int(model_num) - 1], optimizer)

    elif model_num == '2':
        model = SimpleInstanceSeg(num_classes=11).to(DEVICE)
        for optimizer_name in optimizer_names:
            optimizer = get_optimizer(model, optimizer_name)
            train(model, f'{model_names[int(model_num) - 1]}_{optimizer_name}', optimizer)

    else:
        print('Invalid choice')
        sys.exit(1)
    
    utils.show_results(model, dataloaders["test"], DEVICE, THRESHOLD)