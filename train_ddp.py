import argparse
from datetime import datetime
import os

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from model import HAT
from data import SQLDataset


def train(gpu, args):
    # setup
    rank = args.nr * args.gpus + gpu
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        world_size=args.world_size,
        rank=rank
    )

    torch.manual_seed(0)
    scale = 2
    patch_size = 64
    model = HAT(upscale=scale, in_chans=3, img_size=patch_size, window_size=16,
                compress_ratio=3, squeeze_factor=30, conv_scale=0.01, overlap_ratio=0.5,
                img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=180,
                num_heads=[6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6],
                mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    torch.cuda.set_device(gpu)
    model.cuda(gpu)

    # Wrapper around our model to handle parallel training
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    # load
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}
    model.load_state_dict(
        torch.load('model-l-ddp.pth', map_location=map_location))

    # define loss function (criterion) and optimizer
    criterion = nn.L1Loss().cuda(gpu)
    optimizer = torch.optim.Adam(model.parameters(), 2e-4)

    # Data loading code
    batch_size = 2
    train_dataset = SQLDataset('/kaggle/input/ani2k-p64-x2/train_p64_x2.db', hflip=False, vflip=False, rotate=False)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=False,
                                               pin_memory=True,
                                               sampler=train_sampler)
    if gpu == 0:
        print(len(train_dataset))

    start = datetime.now()
    total_step = len(train_loader)
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        for i, (images, labels) in enumerate(train_loader):
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0 and gpu == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                    epoch + 1,
                    args.epochs,
                    i + 1,
                    total_step,
                    loss.item())
                )
            if (i + 1) % 500 == 0 and gpu == 0:
                torch.save(model.state_dict(), 'model.pth')
    if gpu == 0:
        print("Training complete in: " + str(datetime.now() - start))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--nodes', default=1, type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    parser.add_argument('--epochs', default=2, type=int, metavar='N',
                        help='number of total epochs to run')
    args = parser.parse_args()

    args.world_size = args.gpus * args.nodes
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    mp.spawn(train, nprocs=args.gpus, args=(args,))


if __name__ == '__main__':
    main()
