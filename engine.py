import math
import sys
import time

import torch
import torchvision.models.detection.mask_rcnn
import utils
from coco_eval import CocoEvaluator
from coco_utils import get_coco_api_from_dataset

def train_own_model(name, model, optimizer, train_loader, val_loader, device, threshold, num_epochs, writer):
    for epoch in range(num_epochs):
        model.train()
        metric_logger = utils.MetricLogger()
        header = f'Epoch [{epoch}/{num_epochs}]'
        
        for batch_idx, (images, targets) in enumerate(metric_logger.log_every(train_loader, 100, header)):
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()

            if not math.isfinite(loss_value):
                print(f"Loss is {loss_value}, stopping training")
                print(loss_dict)
                sys.exit(1)
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            for name, loss in loss_dict.items():
                writer.add_scalar(f'Loss/{name}', loss.item(), epoch)
            writer.add_scalar('Loss/total', losses.item(), epoch)

            if batch_idx % 200 == 0: 
                utils.tensorboard_save_imgs(model, writer, epoch, batch_idx, val_loader, device, threshold)

            metric_logger.update(**loss_dict)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(device) for img in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                loss_dict = model(images, targets)
                batch_loss = sum(loss for loss in loss_dict.values())
                val_loss += batch_loss.item()
        val_loss /= len(val_loader)
        writer.add_scalar('Validation/Loss', val_loss, epoch)
        
        torch.save(model.state_dict(), f'data/model_checkpoints/{name}_model.pth')
    
    writer.close()
    return model

def train_one_epoch(model, optimizer, data_loader, device, threshold, epoch, writer, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = f"Epoch: [{epoch}]"

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=warmup_factor, total_iters=warmup_iters
        )

    for batch_idx, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        with torch.amp.autocast('cuda', enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        losses_reduced = losses
        loss_value = losses.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        writer.add_scalar("Loss/Total", loss_value, epoch * len(data_loader) + batch_idx)
        for loss_name, loss_val in loss_dict.items():
            writer.add_scalar(f"Loss/{loss_name}", loss_val.item(), epoch * len(data_loader) + batch_idx)

        if batch_idx % 200 == 0: 
            utils.tensorboard_save_imgs(model, writer, epoch, batch_idx, data_loader, device, threshold)
            model.train()

        metric_logger.update(loss=losses_reduced, **loss_dict)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.inference_mode()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"]: output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator