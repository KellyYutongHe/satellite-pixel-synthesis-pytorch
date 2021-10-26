import argparse
import math
import random
import os
import pandas as pd
from PIL import Image

import numpy as np
import torch
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
# from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
from dataset import Naip2SentinelTDataset, Naip2SentinelTPath, MSNSTPDataset
# from calculate_fid import calculate_fid
from distributed import get_rank, synchronize, reduce_loss_dict
from tensor_transforms import convert_to_coord_format
import torchvision.models as models
import tensor_transforms as tt

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch

def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]

def stack_patches(patches, batch_size, resolution, patch_size, channel_size = 3):
    result = torch.zeros(batch_size, channel_size, resolution, resolution)
    n = resolution // patch_size
    for i in range(n):
        for j in range(n):
            result[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[(i,j)]
    return result

def stack_sliding_patches(patches, batch_size, resolution, patch_size, channel_size = 3, device = "cuda"):
    result = torch.zeros(batch_size, channel_size, resolution, resolution).to(device)
    n = resolution // patch_size
    quarter_size = patch_size//4
    for i in range(n):
        for j in range(n):
            result[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[(i,j)]
    for i in range(n):
        for j in range(n-1):
            result[:, :, i*patch_size:(i+1)*patch_size, (j+1)*patch_size-quarter_size:(j+1)*patch_size+quarter_size] += patches[(i,j+0.5)][:,:,:,quarter_size:quarter_size*3]*2
            result[:, :, i*patch_size:(i+1)*patch_size, (j+1)*patch_size-quarter_size:(j+1)*patch_size+quarter_size]/= 3
    for j in range(n):
        for i in range(n-1):
            result[:, :, (i+1)*patch_size-quarter_size:(i+1)*patch_size+quarter_size, j*patch_size:(j+1)*patch_size] += patches[(i+0.5,j)][:,:,quarter_size:quarter_size*3,:]*2
            result[:, :, (i+1)*patch_size-quarter_size:(i+1)*patch_size+quarter_size, j*patch_size:(j+1)*patch_size] /=3
    return result

def test_loader(args, g_ema, device):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    testset = Naip2SentinelTPath(args.test_path, transform=transform, enc_transform=transform,
                                    resolution=args.coords_size, integer_values=args.coords_integer_values)
    test_loader = data.DataLoader(
        testset,
        batch_size=1,
        sampler=data_sampler(testset, shuffle=False, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    pbar = range(len(test_loader))

    loader = sample_data(test_loader)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

#     if args.distributed:
#         g_module = generator.module
#         d_module = discriminator.module

#     else:
#         g_module = generator
#         d_module = discriminator

    with torch.no_grad():
        g_ema.eval()


        for idx in pbar:
            i = idx + args.start_iter

            if i > len(pbar):
                print('Done!')

                break

            highres, lowres_img, highres_img2, img_path = next(loader)
            highres = highres.to(device)
            lowres_img = lowres_img.to(device)
            highres_img2 = highres_img2.to(device)

            real_img, converted = highres[:, :3], highres[:, 3:]

            noise = mixing_noise(1, args.latent, args.mixing, device)

            sample, _ = g_ema(converted, lowres_img, highres_img2, noise)

            utils.save_image(
                sample,
                os.path.join(path, 'cls', args.output_dir, img_path[0].replace("tif", "png")),
                nrow=1,
                normalize=True,
                range=(-1, 1),
            )


def test_patch_loader(args, g_ema, device):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    testset = MSNSTPDataset(args.test_path, transform=transform, enc_transform=transform,
                                    resolution=args.coords_size, crop_size = args.crop_size,
                                    integer_values=args.coords_integer_values)
    test_loader = data.DataLoader(
        testset,
        batch_size=1,
        sampler=data_sampler(testset, shuffle=False, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    pbar = range(len(test_loader))

    loader = sample_data(test_loader)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

#     if args.distributed:
#         g_module = generator.module
#         d_module = discriminator.module

#     else:
#         g_module = generator
#         d_module = discriminator

    with torch.no_grad():
        g_ema.eval()


        for idx in pbar:
            i = idx + args.start_iter

            if i > len(pbar):
                print('Done!')

                break

            sample_patches = {}

            test_data, img_path = next(loader)

            filename = os.path.join(path, 'cls', args.output_dir, img_path[0].replace("tif", "png"))

            if os.path.exists(filename) and i<14000:
                continue
            else:
                for patch_index in test_data.keys():
                    highres, lowres_img, highres_img2, h_start, w_start = test_data[patch_index]
                    highres = highres.to(device)
                    lowres_img = lowres_img.to(device)
                    highres_img2 = highres_img2.to(device)

                    real_img, converted = highres[:, :3], highres[:, 3:]

                    noise = mixing_noise(1, args.latent, args.mixing, device)

                    sample, _ = g_ema(converted, lowres_img, highres_img2, noise, h_start, w_start)

                    sample_patches[patch_index] = sample

                sample = stack_sliding_patches(sample_patches, 1, args.coords_size, args.crop_size)
    #             sample = stack_patches(sample_patches, 1, args.coords_size, args.crop_size)

                utils.save_image(
                    sample,
                    os.path.join(path, 'cls', args.output_dir, img_path[0].replace("tif", "png")),
                    nrow=1,
                    normalize=True,
                    range=(-1, 1),
                )


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--test_path', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="texas_housing_test")
    parser.add_argument('--out_path', type=str, default='.')

    # fid
    parser.add_argument('--fid_samples', type=int, default=50000)
    parser.add_argument('--fid_batch', type=int, default=8)

    # testing
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--start_iter', type=int, default=0)

    # dataset
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--to_crop', action='store_true')
    parser.add_argument('--crop_size', type=int, default=256)
    parser.add_argument('--coords_size', type=int, default=256)
    parser.add_argument('--enc_res', type=int, default=256)

    # Generator params
    parser.add_argument('--Generator', type=str, default='CIPSAtt')
    parser.add_argument('--coords_integer_values', action='store_true')
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--fc_dim', type=int, default=256)
    parser.add_argument('--latent', type=int, default=256)
    parser.add_argument('--linear_dim', type=int, default=256)
    parser.add_argument('--activation', type=str, default=None)
    parser.add_argument('--channel_multiplier', type=int, default=2)
    parser.add_argument('--mixing', type=float, default=0.)
    parser.add_argument('--g_reg_every', type=int, default=4)
    parser.add_argument('--n_mlp', type=int, default=3)

    # Discriminator params
    parser.add_argument('--img2dis',  action='store_true')

    args = parser.parse_args()
    path = args.out_path

    Generator = getattr(model, args.Generator)
    print('Generator', Generator)

    os.makedirs(os.path.join(path, 'cls', args.output_dir), exist_ok=True)
    args.path_fid = os.path.join(path, 'application', args.output_dir)
    os.makedirs(args.path_fid, exist_ok=True)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1
    print("Using:", n_gpu, "GPUs")

#     if args.distributed:
#         print("Parallelized")
#         torch.cuda.set_device(args.local_rank)
#         torch.distributed.init_process_group(backend='nccl', init_method='env://')
#         synchronize()

#     args.n_mlp = 1
    args.dis_input_size = 9 if args.img2dis else 12
    print('img2dis', args.img2dis, 'dis_input_size', args.dis_input_size)

    n_scales = int(math.log(args.size//args.crop_size, 2)) + 1
    print('n_scales', n_scales)

    g_ema = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                      activation=args.activation, linear_size = args.linear_dim,  crop_size = args.crop_size, channel_multiplier=args.channel_multiplier,
                      ).to(device)
    g_ema.eval()

    if args.ckpt is not None:
        print('load model:', args.ckpt)

        ckpt = torch.load(args.ckpt)

#         generator.load_state_dict(ckpt['g'])
#         discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        del ckpt
        torch.cuda.empty_cache()
    else:
        raise Exception("You need to include a checkpoint file for your model")

#     if args.distributed:
#         generator = nn.parallel.DistributedDataParallel(
#             generator,
#             device_ids=[args.local_rank],
#             output_device=args.local_rank,
#             broadcast_buffers=False,
#         )

#         discriminator = nn.parallel.DistributedDataParallel(
#             discriminator,
#             device_ids=[args.local_rank],
#             output_device=args.local_rank,
#             broadcast_buffers=False,
#         )
#
#         encoder = nn.parallel.DistributedDataParallel(
#             encoder,
#             device_ids=[args.local_rank],
#             output_device=args.local_rank,
#             broadcast_buffers=False,
#         )

# test_loader(args, g_ema, device)
# test_patch_loader(args, g_ema, device)
