__all__ = [
           'CIPSAtt', 'CIPSAttProj'
           ]

import math

import torch
from torch import nn
import torch.nn.functional as F

from .blocks import ConstantInput, LFF, StyledConv, ToRGB, PixelNorm, EqualLinear, StyledResBlock, ResBlock, ToRGBNoMod, EqualConv2d, StyledConvNoNoise, ConvLayer, Self_Attn, ConstantInputPatch


class CIPSAtt(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, linear_size = 512, channel_multiplier=2, coord_size = 3, **kwargs):
        super(CIPSAtt, self).__init__()

        self.size = size
        demodulate = True
        self.demodulate = demodulate
        self.coord_size = coord_size
        self.lff = LFF(hidden_size, coord_size = coord_size)
        self.emb = ConstantInput(hidden_size, size=size)


        self.channels = {
            0: linear_size,
            1: linear_size,
            2: linear_size,
            3: linear_size,
            4: int(linear_size/2) * channel_multiplier,
            5: int(linear_size/4) * channel_multiplier,
            6: int(linear_size/8) * channel_multiplier,
            7: int(linear_size/8) * channel_multiplier,
            8: int(linear_size/16) * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])
        self.project1 = ConvLayer(6, in_channels, 3, downsample = True)
        self.project2 = ConvLayer(in_channels, in_channels, 3, downsample = True)
        self.att = Self_Attn(in_channels, "relu")
        self.up_project1 = ConvLayer(in_channels, in_channels, 3, upsample = True)
        self.up_project2 = ConvLayer(in_channels, in_channels, 3, upsample = True)
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledConv(in_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.linears.append(StyledConv(out_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.to_rgbs.append(ToRGB(out_channels, style_dim, upsample=False))

            in_channels = out_channels

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

    def forward(self,
                coords,
                latent,
                input2,
                noise,
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                ):

        noise = noise[0]

        if truncation < 1:
            noise = truncation_latent + truncation * (noise - truncation_latent)

        if not input_is_latent:
            noise = self.style(noise)

        x = self.lff(coords)

        batch_size, _, w, h = coords.shape
        emb = self.emb(x)

        x = torch.cat([x, emb], 1)
        x = self.conv1(x, noise)

        if len(input2.shape) == 1:
            input2 = input2.unsqueeze(0)
            latent = latent.unsqueeze(0)

        latent = torch.cat([latent, input2], 1)
        latent = self.project1(latent)
#         print(latent.shape)
        latent = self.project2(latent)
        latent, _ = self.att(latent)
#         print(latent.shape)
        latent = self.up_project1(latent)
#         print(latent.shape)
        latent = self.up_project2(latent)
#         print(latent.shape)

        x = x + latent

        rgb = 0

        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, noise)

            rgb = self.to_rgbs[i](x, noise, rgb)

        if return_latents:
            return rgb, noise
        else:
            return rgb, None

class CIPSAttProj(nn.Module):
    def __init__(self, size=256, hidden_size=512, n_mlp=8, style_dim=512, lr_mlp=0.01,
                 activation=None, linear_size = 512, channel_multiplier=2, coord_size = 3, crop_size = 64, device = "cuda",
                 n_att=3, **kwargs):
        super(CIPSAttProj, self).__init__()

        self.size = size
        self.crop_size = crop_size
        self.n_att = n_att
        demodulate = True
        self.demodulate = demodulate
        self.coord_size = coord_size
        self.lff = LFF(hidden_size, coord_size = coord_size)
        self.emb = ConstantInputPatch(hidden_size, size=size, device = device)


        self.channels = {
            0: linear_size,
            1: linear_size,
            2: linear_size,
            3: linear_size,
            4: int(linear_size/2) * channel_multiplier,
            5: int(linear_size/4) * channel_multiplier,
            6: int(linear_size/8) * channel_multiplier,
            7: int(linear_size/8) * channel_multiplier,
            8: int(linear_size/16) * channel_multiplier,
        }

        multiplier = 2
        in_channels = int(self.channels[0])
        self.project_linear = ConvLayer(6, in_channels, 1)

        self.project = nn.ModuleList()
        self.proj_channels = {
            0: linear_size,
            1: linear_size//2,
            2: linear_size//2,
            3: linear_size
        }
        for k in range(3):
            self.project.append(ConvLayer(self.proj_channels[k], self.proj_channels[k+1], 3))
        self.att = Self_Attn(in_channels, "relu")
#         self.att = nn.ModuleList()
#         for k in range(self.n_att):
#             self.att.append(Self_Attn(in_channels, "relu"))
        self.conv1 = StyledConv(int(multiplier*hidden_size), in_channels, 1, style_dim, demodulate=demodulate,
                                activation=activation)

        self.linears = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.log_size = int(math.log(size, 2))

        self.n_intermediate = self.log_size - 1
        self.to_rgb_stride = 2
        for i in range(0, self.log_size - 1):
            out_channels = self.channels[i]
            self.linears.append(StyledConv(in_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.linears.append(StyledConv(out_channels, out_channels, 1, style_dim,
                                           demodulate=demodulate, activation=activation))
            self.to_rgbs.append(ToRGB(out_channels, style_dim, upsample=False))

            in_channels = out_channels

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

    def forward(self,
                coords,
                latent,
                input2,
                noise,
                h_start,
                w_start,
                return_latents=False,
                truncation=1,
                truncation_latent=None,
                input_is_latent=False,
                ):

        noise = noise[0]

        if truncation < 1:
            noise = truncation_latent + truncation * (noise - truncation_latent)

        if not input_is_latent:
            noise = self.style(noise)

        x = self.lff(coords)

        batch_size, _, w, h = coords.shape
        emb = self.emb(x, h_start, w_start, crop_size = self.crop_size)
#         print(emb.shape)

        x = torch.cat([x, emb], 1)
        x = self.conv1(x, noise)

        if len(input2.shape) == 1:
            input2 = input2.unsqueeze(0)
            latent = latent.unsqueeze(0)

        latent = torch.cat([latent, input2], 1)
        latent0 = self.project_linear(latent)
        latent = latent0
        for k in range(3):
            latent = self.project[k](latent)
        latent, _ = self.att(latent)
        latent = latent + latent0
        x = x + latent

        rgb = 0

        for i in range(self.n_intermediate):
            for j in range(self.to_rgb_stride):
                x = self.linears[i*self.to_rgb_stride + j](x, noise)

            rgb = self.to_rgbs[i](x, noise, rgb)

        if return_latents:
            return rgb, noise
        else:
            return rgb, None
