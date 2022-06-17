import argparse
import math
import random
import os

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import model
from model.loss import SSIM
from dataset import PatchNSTDataset, MSNSTDataset
from distributed import get_rank, synchronize, reduce_loss_dict
from tensor_transforms import convert_to_coord_format
import torchvision.models as models


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)

def stack_patches(patches, batch_size, resolution, patch_size, channel_size = 3):
    result = torch.zeros(batch_size, channel_size, resolution, resolution)
    n = resolution // patch_size
    for i in range(n):
            for j in range(n):
#                 print(patches[(i,j)].shape)
                result[:, :, i*patch_size:(i+1)*patch_size, j*patch_size:(j+1)*patch_size] = patches[(i,j)]
    return result


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha = 1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).view(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss

def g_rec_loss(fake, real):
    loss = F.l1_loss(fake, real)
#     loss = F.mse_loss(fake, real, reduction='sum') / fake.size(0)

    return loss


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


def train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, test_data, device):
#     requires_grad(encoder, False)
    loader = sample_data(loader)

    ssim = SSIM()

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.01)

    mean_path_length = 0

    r1_loss = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))

    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print('Done!')

            break

        # data = next(loader)
        # key = np.random.randint(n_scales)
        # real_stack = data[key].to(device)
        highres, lowres_img, highres_img2, h_start, w_start = next(loader)
        highres = highres.to(device)
        lowres_img = lowres_img.to(device)
        highres_img2 = highres_img2.to(device)

        real_img, converted = highres[:, :3], highres[:, 3:]

        # Training Discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        if i == 0:
            utils.save_image(
                            lowres_img,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images/train_lowres.png'),
                            nrow=int(lowres_img.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

            utils.save_image(
                highres_img2,
                os.path.join(
                    path,
                    f'outputs/{args.output_dir}/images/train_highres2.png'),
                nrow=int(highres_img2.size(0) ** 0.5),
                normalize=True,
                range=(-1, 1),
            )

            utils.save_image(
                real_img,
                os.path.join(
                    path,
                    f'outputs/{args.output_dir}/images/train_real_patch.png'),
                nrow=int(real_img.size(0) ** 0.5),
                normalize=True,
                range=(-1, 1),
            )

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)
#         lowres_embedding = encoder(lowres_img).squeeze()
#         highres2_embedding = encoder(highres_img2).squeeze()
#         print(lowres_embedding.shape)

        fake_img, _ = generator(converted, lowres_img, highres_img2, noise, h_start, w_start)
        fake = fake_img if args.img2dis else torch.cat([fake_img, converted], 1)
        fake_input_d = torch.cat((fake, lowres_img, highres_img2), 1)
        fake_pred = discriminator(fake_input_d)

        real = real_img if args.img2dis else highres
        real_input_d = torch.cat((real, lowres_img, highres_img2), 1)
        real_pred = discriminator(real_input_d)
        d_loss = d_logistic_loss(real_pred, fake_pred)

        loss_dict['d'] = d_loss
        loss_dict['real_score'] = real_pred.mean()
        loss_dict['fake_score'] = fake_pred.mean()

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_input_d.requires_grad = True
            real_pred = discriminator(real_input_d)
            r1_loss = d_r1_loss(real_pred, real_input_d)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()

            d_optim.step()

        loss_dict['r1'] = r1_loss

        # Training Generator
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.latent, args.mixing, device)

        fake_img, _ = generator(converted, lowres_img, highres_img2, noise, h_start, w_start)
        fake = fake_img if args.img2dis else torch.cat([fake_img, converted], 1)
        fake_input_g = torch.cat((fake, lowres_img, highres_img2), 1)
        fake_pred = discriminator(fake_input_g)
        g_gan_loss = g_nonsaturating_loss(fake_pred)
        real_img = highres[:, :3]
        rec_loss = g_rec_loss(fake_img, real_img)
        ssim_loss = 1.0 - ssim(fake_img, real_img)
        g_loss = g_gan_loss + args.l1_lambda*rec_loss + args.ssim_lambda*ssim_loss


        loss_dict['g_gan'] = g_gan_loss
        loss_dict['g_rec'] = rec_loss
        loss_dict['g_ssim'] = ssim_loss
        loss_dict['g'] = g_loss

        generator.zero_grad()
        g_loss.backward()
        g_optim.step()

        loss_dict['path'] = path_loss
        loss_dict['path_length'] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = reduce_loss_dict(loss_dict)

        d_loss_val = loss_reduced['d'].mean().item()
        g_loss_val = loss_reduced['g'].mean().item()
        g_gan_loss_val = loss_reduced['g_gan'].mean().item()
        g_rec_loss_val = loss_reduced['g_rec'].mean().item()
        g_ssim_loss_val = loss_reduced['g_ssim'].mean().item()
        r1_val = loss_reduced['r1'].mean().item()
        path_loss_val = loss_reduced['path'].mean().item()
        real_score_val = loss_reduced['real_score'].mean().item()
        fake_score_val = loss_reduced['fake_score'].mean().item()
        path_length_val = loss_reduced['path_length'].mean().item()

        if get_rank() == 0:
            pbar.set_description(
                (
                    f'd: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1: {r1_val:.4f}; rec: {g_rec_loss_val:.4f}; ssim: {g_ssim_loss_val:.4f};'
                )
            )

            if i % 10 == 0:
                writer.add_scalar("Generator", g_loss_val, i)
                writer.add_scalar("Discriminator", d_loss_val, i)
                writer.add_scalar("R1", r1_val, i)
                writer.add_scalar("REC", g_rec_loss_val, i)
                writer.add_scalar("SSIM", g_ssim_loss_val, i)
                writer.add_scalar("Path Length Regularization", path_loss_val, i)
                writer.add_scalar("Mean Path Length", mean_path_length, i)
                writer.add_scalar("Real Score", real_score_val, i)
                writer.add_scalar("Fake Score", fake_score_val, i)
                writer.add_scalar("Path Length", path_length_val, i)

            if i % 100 == 0:
                with torch.no_grad():
                    g_ema.eval()

                    sample_patches = {}
                    if i == 0:
                        low_patches = {}
                        high2_patches = {}
                        real_pathces = {}

                    for patch_index in test_data.keys():
                        highres, lowres_img, highres_img2, h_start, w_start = test_data[patch_index]
                        highres = highres.to(device)
                        lowres_img = lowres_img.to(device)
                        highres_img2 = highres_img2.to(device)

                        real_img, converted = highres[:, :3], highres[:, 3:]

                        sample, _ = g_ema(converted, lowres_img, highres_img2, [sample_z], h_start, w_start)

                        sample_patches[patch_index] = sample

                        if i == 0:
                            low_patches[patch_index] = lowres_img
                            high2_patches[patch_index] = highres_img2
                            real_pathces[patch_index] = real_img

                    sample = stack_patches(sample_patches, args.n_sample, args.coords_size, args.crop_size)

                    utils.save_image(
                        sample,
                        os.path.join(path, 'outputs', args.output_dir, 'images', f'{str(i).zfill(6)}.png'),
                        nrow=int(args.n_sample ** 0.5),
                        normalize=True,
                        range=(-1, 1),
                    )

                    if i == 0:
                        lowres_img = stack_patches(low_patches, args.n_sample, args.coords_size, args.crop_size)
                        highres_img2 = stack_patches(high2_patches, args.n_sample, args.coords_size, args.crop_size)
                        real_img = stack_patches(real_pathces, args.n_sample, args.coords_size, args.crop_size)
                        utils.save_image(
                            lowres_img,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images/lowres.png'),
                            nrow=int(lowres_img.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            highres_img2,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images/highres2.png'),
                            nrow=int(highres_img2.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

                        utils.save_image(
                            real_img,
                            os.path.join(
                                path,
                                f'outputs/{args.output_dir}/images/real_patch.png'),
                            nrow=int(real_img.size(0) ** 0.5),
                            normalize=True,
                            range=(-1, 1),
                        )

            if i % args.save_checkpoint_frequency == 0:
                torch.save(
                    {
                        'g': g_module.state_dict(),
                        'd': d_module.state_dict(),
                        'g_ema': g_ema.state_dict(),
                        'g_optim': g_optim.state_dict(),
                        'd_optim': d_optim.state_dict(),
                    },
                    os.path.join(
                        path,
                        f'outputs/{args.output_dir}/checkpoints/{str(i).zfill(6)}.pt'),
                )
#             if i % (args.save_checkpoint_frequency*10) == 0 and i > 0:
#                 cur_metrics = calculate_fid(g_ema, fid_dataset=fid_dataset, bs=args.fid_batch, size=args.coords_size,
#                                             num_batches=args.fid_samples//args.fid_batch, latent_size=args.latent,
#                                             save_dir=args.path_fid, integer_values=args.coords_integer_values)
#                 writer.add_scalar("fid", cur_metrics['frechet_inception_distance'], i)
#                 print(i, "fid",  cur_metrics['frechet_inception_distance'])


if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()

    parser.add_argument('--path', type=str, default="")
    parser.add_argument('--test_path', type=str, default="")
    parser.add_argument('--output_dir', type=str, default="texas_test")
    parser.add_argument('--out_path', type=str, default='.')

    # fid
    parser.add_argument('--fid_samples', type=int, default=50000)
    parser.add_argument('--fid_batch', type=int, default=8)

    # training
    parser.add_argument('--iter', type=int, default=1200000)
    parser.add_argument('--n_sample', type=int, default=8)
    parser.add_argument('--generate_by_one', action='store_true')
    parser.add_argument('--ckpt', type=str, default=None)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--l1_lambda', type=float, default=100)
    parser.add_argument('--ssim_lambda', type=float, default=0)
    parser.add_argument('--save_checkpoint_frequency', type=int, default=2000)

    # dataset
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--to_crop', action='store_true')
    parser.add_argument('--crop_size', type=int, default=64)
    parser.add_argument('--coords_size', type=int, default=256)
    parser.add_argument('--enc_res', type=int, default=224)

    # Generator params
    parser.add_argument('--Generator', type=str, default='CIPSAttProj')
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
    parser.add_argument('--n_att', type=int, default=3)


    # Discriminator params
    parser.add_argument('--Discriminator', type=str, default='Discriminator')
    parser.add_argument('--d_reg_every', type=int, default=16)
    parser.add_argument('--r1', type=float, default=10)
    parser.add_argument('--img2dis',  action='store_true')
    parser.add_argument('--n_first_layers', type=int, default=0)

    args = parser.parse_args()
    path = args.out_path

    Generator = getattr(model, args.Generator)
    print('Generator', Generator)
    Discriminator = getattr(model, args.Discriminator)
    print('Discriminator', Discriminator)

    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(path, 'outputs', args.output_dir, 'checkpoints'), exist_ok=True)
    args.logdir = os.path.join(path, 'tensorboard', args.output_dir)
    os.makedirs(args.logdir, exist_ok=True)
    args.path_fid = os.path.join(path, 'fid', args.output_dir)
    os.makedirs(args.path_fid, exist_ok=True)

    n_gpu = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
    args.distributed = n_gpu > 1
    print("Using:", n_gpu, "GPUs")

    if args.distributed:
        print("Parallelized")
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')
        synchronize()

#     args.n_mlp = 3
    args.dis_input_size = 9 if args.img2dis else 12
    print('img2dis', args.img2dis, 'dis_input_size', args.dis_input_size)

    args.start_iter = 0
    n_scales = int(math.log(args.size//args.crop_size, 2)) + 1
    print('n_scales', n_scales)

    generator = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                          activation=args.activation, linear_size = args.linear_dim, channel_multiplier=args.channel_multiplier,
                          crop_size=args.crop_size, device = device, n_att = args.n_att).to(device)

    print('generator N params', sum(p.numel() for p in generator.parameters() if p.requires_grad))
    discriminator = Discriminator(
        size=args.crop_size, channel_multiplier=args.channel_multiplier, n_scales=n_scales, input_size=args.dis_input_size,
        n_first_layers=args.n_first_layers,
    ).to(device)
    g_ema = Generator(size=args.size, hidden_size=args.fc_dim, style_dim=args.latent, n_mlp=args.n_mlp,
                      activation=args.activation, linear_size = args.linear_dim, channel_multiplier=args.channel_multiplier,
                      crop_size=args.crop_size, device = device, n_att = args.n_att
                      ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

#     resnet18 = models.resnet18(pretrained=True).to(device)
#     modules = list(resnet18.children())[:-1]
#     encoder = nn.Sequential(*modules)

    g_optim = optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )
    d_optim = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    if args.ckpt is not None:
        print('load model:', args.ckpt)

        ckpt = torch.load(args.ckpt)

        try:
            ckpt_name = os.path.basename(args.ckpt)
            args.start_iter = int(os.path.splitext(ckpt_name)[0])

        except ValueError:
            pass

        generator.load_state_dict(ckpt['g'])
        discriminator.load_state_dict(ckpt['d'])
        g_ema.load_state_dict(ckpt['g_ema'])

        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])

        del ckpt
        torch.cuda.empty_cache()

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

#         encoder = nn.parallel.DistributedDataParallel(
#             encoder,
#             device_ids=[args.local_rank],
#             output_device=args.local_rank,
#             broadcast_buffers=False,
#         )

    enc_transform = transforms.Compose(
        [
            # transforms.Resize(256),
            # transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True),
        ]
    )
    # transform_fid = transforms.Compose([
    #                                    transforms.ToTensor(),
    #                                    transforms.Lambda(lambda x: x.mul_(255.).byte())])
    # dataset = MultiScaleDataset(args.path, transform=transform, resolution=args.coords_size, crop_size=args.crop,
    #                             integer_values=args.coords_integer_values, to_crop=args.to_crop)
    dataset = PatchNSTDataset(args.path, transform=transform, enc_transform=enc_transform,
                                    resolution=args.coords_size, crop_size = args.crop_size,
                                    integer_values=args.coords_integer_values)
    testset = MSNSTDataset(args.test_path, transform=transform, enc_transform=enc_transform,
                                    resolution=args.coords_size, crop_size = args.crop_size,
                                    integer_values=args.coords_integer_values)
    # fid_dataset = ImageDataset(args.path, transform=transform_fid, resolution=args.coords_size, to_crop=args.to_crop)
    # fid_dataset.length = args.fid_samples
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = data.DataLoader(
        testset,
        batch_size=args.n_sample,
        sampler=data_sampler(testset, shuffle=False, distributed=args.distributed),
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    test_data = iter(test_loader).next()
    del testset
    del test_loader
#     print(test_data[0].shape)

    writer = SummaryWriter(log_dir=args.logdir)

    train(args, loader, generator, discriminator, g_optim, d_optim, g_ema, test_data, device)
