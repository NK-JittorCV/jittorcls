"""
Licensed under the Apache License, Version 2.0.
Modify from https://github.com/Jittor-Image-Models/Jittor-Image-Models
"""

import argparse
import numpy as np
from tqdm import tqdm
import jittor as jt
from jittor import nn
from jittor import transform
from jvcore.scheduler import CosineAnnealingLR
import logging

from jvcore.models import create_model
from jvcore.data import create_dataset
jt.flags.use_cuda = 1


def get_train_transforms():
    return transform.Compose([
        transform.RandomCropAndResize((224, 224)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


def get_valid_transforms():
    return transform.Compose([
        transform.Resize(256),
        transform.CenterCrop(224),
        transform.ToTensor(),
        transform.ImageNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])


@jt.enable_grad()
@jt.single_process_scope()
def train(model, train_loader, criterion, optimizer, epoch, accum_iter, scheduler):
    model.train()
    # total_acc = 0
    # total_num = 0
    losses = []

    pbar = tqdm(train_loader, desc=f'Epoch {epoch} [TRAIN]', maxinterval=len(train_loader))
    for i, (images, labels) in enumerate(pbar):
        output = model(images)
        if labels.requires_grad:
            labels.stop_grad()
        loss = criterion(output, labels)

        optimizer.backward(loss)
        if (i + 1) % accum_iter == 0 or i + 1 == len(train_loader):
            optimizer.step(loss)

        # pred = np.argmax(output.numpy(), axis=1)
        # acc = np.sum(pred == labels.numpy())
        # total_acc += acc
        # total_num += labels.shape[0]
        losses.append(loss.numpy()[0])
        pbar.update(1)
        pbar.set_description(f'Epoch {epoch} loss={sum(losses) / len(losses):.2f}')
    scheduler.step()

@jt.no_grad()
@jt.single_process_scope()
def val(model, val_loader, epoch):
    model.eval()
    total_acc = 0
    total_num = 0
    pbar = tqdm(val_loader, desc=f'[VALID]', maxinterval=len(val_loader))
    for batch_idx, (inputs, targets) in enumerate(val_loader):
        batch_size = inputs.shape[0]
        outputs = model(inputs)
        pred = np.argmax(outputs.numpy(), axis=1)
        acc = np.sum(targets.numpy()==pred)
        total_acc += acc
        total_num += batch_size
        acc = acc / batch_size
        # total_acc = jt.int64(total_acc)
        # total_num = jt.int64(total_num)
        pbar.update(1)
        pbar.set_description(f'Test Epoch: {epoch} [{batch_idx}/{len(val_loader)}]\tAcc: {acc:.6f}')
    # if jt.in_mpi:
    #     total_acc = total_acc.mpi_all_reduce()
    #     total_num = total_num.mpi_all_reduce()
    # if jt.rank == 0:
    #     return total_acc.numpy() / total_num.numpy()
    return total_acc / total_num
    

if __name__ == '__main__':
    jt.set_global_seed(648)
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=float, default=0.9)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--accum_iter', type=int, default=1)
    parser.add_argument('--log_info', type=str, default='Log')
    parser.add_argument('--save_path', type=str, default='')
    parser.add_argument('--val_time', type=int, default=1)
    parser.add_argument('--model', type=str, default='p2t_tiny')
    parser.add_argument('--dataset', type=str, default='ImageNet')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--num_workers', type=int, default=32)
    args = parser.parse_args()

    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)
    handler = logging.FileHandler(args.log_info+".log")
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    train_loader = create_dataset(args.dataset, root=args.data_root, split='train', transform=get_train_transforms(), 
                                batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    val_loader = create_dataset(args.dataset, root=args.data_root, split='validation', transform=get_valid_transforms(), 
                                batch_size=args.batch_size, shuffle=args.shuffle, num_workers=args.num_workers)

    model = create_model(args.model, True)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = nn.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, args.gamma)

    for epoch in range(args.epochs):
        train(model, train_loader, criterion, optimizer, epoch, args.accum_iter, scheduler)
        if (epoch + 1) % args.val_time == 0:
            acc = val(model, val_loader, epoch)
            print('epoch:', epoch, '\tacc:', acc)
    if jt.rank == 0:
        model.save(args.save_path)
