from io import BytesIO
import math

import lmdb
from PIL import Image
from torch.utils.data import Dataset
import torch
import numpy as np

import tensor_transforms as tt
import pandas as pd


class Naip2SentinelTDataset(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution, integer_values):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.integer_values = integer_values
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.naip2016 = np.asarray(self.data_info.iloc[:, 1])
        self.naip2018 = np.asarray(self.data_info.iloc[:, 2])
        self.naip = np.concatenate((self.naip2016, self.naip2018))
        self.sentinel2016 = np.asarray(self.data_info.iloc[:, 3])
        self.sentinel2018 = np.asarray(self.data_info.iloc[:, 4])
        self.sentinel = np.concatenate((self.sentinel2016, self.sentinel2018))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.time = np.concatenate((np.array([0]*data_len),np.array([1]*data_len)))
        self.data_len = len(self.sentinel)
        self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)
        # self.crop = tt.RandomCrop(resolution)
#         print(self.naip.shape, self.sentinel.shape, self.time.shape)

    def __getitem__(self, index):
        # Get image name from the pandas df
        naip = self.naip[index]
        sentinel = self.sentinel[index]
        t = self.time[index]
        if t == 1:
            naip2 = self.naip[index-self.house_count]
        else:
            naip2 = self.naip[index+self.house_count]
        # Open image
        naip = Image.open(naip).convert('RGB')
        naip2 = Image.open(naip2).convert('RGB')
        sentinel = Image.open(sentinel).convert('RGB')
#         im1 = naip.save("naip.jpg")
#         im2 = sentinel.save("sentinel.jpg")
        # Transform the image
        naip = self.transform(naip)
        naip2 = self.enc_transform(naip2)
        sentinel = self.enc_transform(sentinel)
#         print(naip.shape, self.coords[t].shape)
#         naip = torch.cat([naip, self.coords[t]], 1).squeeze(0)
        naip = torch.cat([naip, self.coords[t]], 0)

        # naip = self.crop(sentinel.unsqueeze(0)).squeeze(0)
        # sentinel = self.crop(sentinel.unsqueeze(0)).squeeze(0)

#         print(naip.shape, sentinel.shape)

        return (naip, sentinel, naip2)

    def __len__(self):
        return self.data_len


class Naip2SentinelTPath(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution, integer_values):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.integer_values = integer_values
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.naip2016 = np.asarray(self.data_info.iloc[:, 1])
        self.naip2018 = np.asarray(self.data_info.iloc[:, 2])
        self.naip = np.concatenate((self.naip2016, self.naip2018))
        self.sentinel2016 = np.asarray(self.data_info.iloc[:, 3])
        self.sentinel2018 = np.asarray(self.data_info.iloc[:, 4])
        self.sentinel = np.concatenate((self.sentinel2016, self.sentinel2018))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.time = np.concatenate((np.array([0]*data_len),np.array([1]*data_len)))
        self.data_len = len(self.sentinel)
        self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)
        # self.crop = tt.RandomCrop(resolution)
#         print(self.naip.shape, self.sentinel.shape, self.time.shape)

    def __getitem__(self, index):
        # Get image name from the pandas df
        naip = self.naip[index]
        path = naip.split("/")[-1]
        sentinel = self.sentinel[index]
        t = self.time[index]
        if t == 1:
            naip2 = self.naip[index-self.house_count]
        else:
            naip2 = self.naip[index+self.house_count]
        # Open image
        naip = Image.open(naip).convert('RGB')
        naip2 = Image.open(naip2).convert('RGB')
        sentinel = Image.open(sentinel).convert('RGB')
        # Transform the image
        naip = self.transform(naip)
        naip2 = self.enc_transform(naip2)
        sentinel = self.enc_transform(sentinel)
        naip = torch.cat([naip, self.coords[t]], 0)

        return (naip, sentinel, naip2, path)

    def __len__(self):
        return self.data_len

class PatchNSTDataset(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution=256, crop_size=64, integer_values=False):
        #crop for better fitting into the memory
        self.crop_size = crop_size
#         self.n = resolution // crop_size
#         self.log_size = int(math.log(self.n, 2))
        self.crop = tt.RandomCropDim3(crop_size)
#         self.crop_resolution = tt.RandomCrop(resolution)
#         self.to_crop = to_crop
        self.resolution = resolution

        self.integer_values = integer_values
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.naip2016 = np.asarray(self.data_info.iloc[:, 1])
        self.naip2018 = np.asarray(self.data_info.iloc[:, 2])
        self.naip = np.concatenate((self.naip2016, self.naip2018))
        self.sentinel2016 = np.asarray(self.data_info.iloc[:, 3])
        self.sentinel2018 = np.asarray(self.data_info.iloc[:, 4])
        self.sentinel = np.concatenate((self.sentinel2016, self.sentinel2018))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.time = np.concatenate((np.array([0]*data_len),np.array([1]*data_len)))
        self.data_len = len(self.sentinel)
        self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        naip = self.naip[index]
        path = naip.split("/")[-1]
        sentinel = self.sentinel[index]
        t = self.time[index]
        if t == 1:
            naip2 = self.naip[index-self.house_count]
        else:
            naip2 = self.naip[index+self.house_count]
        # Open image
        naip = Image.open(naip).convert('RGB')
        naip2 = Image.open(naip2).convert('RGB')
        sentinel = Image.open(sentinel).convert('RGB')
        # Transform the image
        naip = self.transform(naip)
        naip2 = self.enc_transform(naip2)
        sentinel = self.enc_transform(sentinel)
        naip = torch.cat([naip, self.coords[t]], 0)
#         print(naip.shape)

        naip, h_start, w_start = self.crop(naip)
        naip2 = tt.patch_crop_dim3(naip2, h_start, w_start, self.crop_size)
        sentinel = tt.patch_crop_dim3(sentinel, h_start, w_start, self.crop_size)
#         print(naip.shape)

        return (naip, sentinel, naip2, h_start, w_start)

class MSNSTDataset(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution=256, crop_size=64, integer_values=False):
        #crop for better fitting into the memory
        self.crop_size = crop_size
        self.n = resolution // crop_size
        self.resolution = resolution

        self.integer_values = integer_values
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.naip2016 = np.asarray(self.data_info.iloc[:, 1])
        self.naip2018 = np.asarray(self.data_info.iloc[:, 2])
        self.naip = np.concatenate((self.naip2016, self.naip2018))
        self.sentinel2016 = np.asarray(self.data_info.iloc[:, 3])
        self.sentinel2018 = np.asarray(self.data_info.iloc[:, 4])
        self.sentinel = np.concatenate((self.sentinel2016, self.sentinel2018))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.time = np.concatenate((np.array([0]*data_len),np.array([1]*data_len)))
        self.data_len = len(self.sentinel)
        self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        data = {}
        naip = self.naip[index]
        path = naip.split("/")[-1]
        sentinel = self.sentinel[index]
        t = self.time[index]
        if t == 1:
            naip2 = self.naip[index-self.house_count]
        else:
            naip2 = self.naip[index+self.house_count]
        # Open image
        naip = Image.open(naip).convert('RGB')
        naip2 = Image.open(naip2).convert('RGB')
        sentinel = Image.open(sentinel).convert('RGB')
        # Transform the image
        naip = self.transform(naip)
        naip2 = self.enc_transform(naip2)
        sentinel = self.enc_transform(sentinel)
        naip = torch.cat([naip, self.coords[t]], 0)
#         print(naip.shape)

        for i in range(self.n):
            for j in range(self.n):
                naip_ij = tt.patch_crop_dim3(naip, i*self.crop_size, j*self.crop_size, self.crop_size)
                naip2_ij = tt.patch_crop_dim3(naip2, i*self.crop_size, j*self.crop_size, self.crop_size)
                sentinel_ij = tt.patch_crop_dim3(sentinel, i*self.crop_size, j*self.crop_size, self.crop_size)
                data[(i,j)] = (naip_ij, sentinel_ij, naip2_ij, i*self.crop_size, j*self.crop_size)
        return data

class MSNSTPDataset(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution=256, crop_size=64, integer_values=False):
        #crop for better fitting into the memory
        self.crop_size = crop_size
        self.n = resolution // crop_size
        self.resolution = resolution

        self.integer_values = integer_values
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.naip2016 = np.asarray(self.data_info.iloc[:, 1])
        self.naip2018 = np.asarray(self.data_info.iloc[:, 2])
        self.naip = np.concatenate((self.naip2016, self.naip2018))
        self.sentinel2016 = np.asarray(self.data_info.iloc[:, 3])
        self.sentinel2018 = np.asarray(self.data_info.iloc[:, 4])
        self.sentinel = np.concatenate((self.sentinel2016, self.sentinel2018))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.time = np.concatenate((np.array([0]*data_len),np.array([1]*data_len)))
        self.data_len = len(self.sentinel)
        self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)

    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        data = {}
        naip = self.naip[index]
        path = naip.split("/")[-1]
        sentinel = self.sentinel[index]
        t = self.time[index]
        if t == 1:
            naip2 = self.naip[index-self.house_count]
        else:
            naip2 = self.naip[index+self.house_count]
        # Open image
        naip_img = Image.open(naip).convert('RGB')
        naip2_img = Image.open(naip2).convert('RGB')
        sentinel_img = Image.open(sentinel).convert('RGB')
        # Transform the image
        naip = self.transform(naip_img)
        naip2 = self.enc_transform(naip2_img)
        sentinel = self.enc_transform(sentinel_img)
        naip_img.close()
        naip2_img.close()
        sentinel_img.close()
        naip = torch.cat([naip, self.coords[t]], 0)
#         print(naip.shape)

        for i in range(self.n):
            for j in range(self.n):
                naip_ij = tt.patch_crop_dim3(naip, i*self.crop_size, j*self.crop_size, self.crop_size)
                naip2_ij = tt.patch_crop_dim3(naip2, i*self.crop_size, j*self.crop_size, self.crop_size)
                sentinel_ij = tt.patch_crop_dim3(sentinel, i*self.crop_size, j*self.crop_size, self.crop_size)
                data[(i,j)] = (naip_ij, sentinel_ij, naip2_ij, i*self.crop_size, j*self.crop_size)
        half_size = self.crop_size//2
        for i in range(self.n):
            for j in range(self.n-1):
                naip_ij = tt.patch_crop_dim3(naip, i*self.crop_size, j*self.crop_size+half_size, self.crop_size)
                naip2_ij = tt.patch_crop_dim3(naip2, i*self.crop_size, j*self.crop_size+half_size, self.crop_size)
                sentinel_ij = tt.patch_crop_dim3(sentinel, i*self.crop_size, j*self.crop_size+half_size, self.crop_size)
                data[(i,j+0.5)] = (naip_ij, sentinel_ij, naip2_ij, i*self.crop_size, j*self.crop_size)
        for j in range(self.n):
            for i in range(self.n-1):
                naip_ij = tt.patch_crop_dim3(naip, i*self.crop_size+half_size, j*self.crop_size, self.crop_size)
                naip2_ij = tt.patch_crop_dim3(naip2, i*self.crop_size+half_size, j*self.crop_size, self.crop_size)
                sentinel_ij = tt.patch_crop_dim3(sentinel, i*self.crop_size+half_size, j*self.crop_size, self.crop_size)
                data[(i+0.5,j)] = (naip_ij, sentinel_ij, naip2_ij, i*self.crop_size, j*self.crop_size)
        return data, path

class FMoWSentinel2(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution, integer_values):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.integer_values = integer_values
        self.resolution = resolution
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.highpast = np.asarray(self.data_info.iloc[:, 1])
        self.highpresent = np.asarray(self.data_info.iloc[:, 2])
        self.high = np.concatenate((self.highpast, self.highpresent))
        self.lowpast = np.asarray(self.data_info.iloc[:, 3])
        self.lowpresent = np.asarray(self.data_info.iloc[:, 4])
        self.low = np.concatenate((self.lowpast, self.lowpresent))
        #process date
        self.pastdate = pd.to_datetime(self.data_info.iloc[:, 5], format="%Y%m%dT%H%M%S")
        self.presentdate = pd.to_datetime(self.data_info.iloc[:, 6], format="%Y%m%dT%H%M%S")
        self.pastdate = (self.pastdate - pd.to_datetime("20150623", format="%Y%m%d")).dt.days
        self.presentdate = (self.presentdate - pd.to_datetime("20150623", format="%Y%m%d")).dt.days
        self.time = np.concatenate((self.pastdate, self.presentdate))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.data_len = len(self.low)
#         self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)
        # self.crop = tt.RandomCrop(resolution)
#         print(self.naip.shape, self.sentinel.shape, self.time.shape)

#     def get_sinusoid_encoding_table(positions, d_hid, T=1000):
#         ''' Sinusoid position encoding table
#         positions: int or list of integer, if int range(positions)'''

#         if isinstance(positions, int):
#             positions = list(range(positions))

#         def cal_angle(position, hid_idx):
#             return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

#         def get_posi_angle_vec(position):
#             return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

#         sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

#         if torch.cuda.is_available():
#             return torch.FloatTensor(sinusoid_table).cuda()
#         else:
#             return torch.FloatTensor(sinusoid_table)

    def __getitem__(self, index):
        # Get image name from the pandas df
        high = self.high[index]
        low = self.low[index]
        t = self.time[index]
        if index >= self.house_count:
            high2 = self.high[index-self.house_count]
        else:
            high2 = self.highpresent[index]
        if index == 0:
            print(high, high2)
        # Open image
        high = Image.open(high).convert('RGB')
        high2 = Image.open(high2).convert('RGB')
        low = Image.open(low).convert('RGB')
        # Transform the image
        high = self.transform(high)
        high2 = self.transform(high2)
        low = self.enc_transform(low)
        coords = tt.convert_to_coord_uneven_t(1, self.resolution, self.resolution, t, integer_values=self.integer_values)
#         print(coords)
        high = torch.cat([high, coords], 0)

        return (high, low, high2)

    def __len__(self):
        return self.data_len

class FMoWSentinel2Path(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution, integer_values):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.integer_values = integer_values
        self.resolution = resolution
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.highpast = np.asarray(self.data_info.iloc[:, 1])
        self.highpresent = np.asarray(self.data_info.iloc[:, 2])
        self.high = np.concatenate((self.highpast, self.highpresent))
        self.lowpast = np.asarray(self.data_info.iloc[:, 3])
        self.lowpresent = np.asarray(self.data_info.iloc[:, 4])
        self.low = np.concatenate((self.lowpast, self.lowpresent))
        #process date
        self.pastdate = pd.to_datetime(self.data_info.iloc[:, 5], format="%Y%m%dT%H%M%S")
        self.presentdate = pd.to_datetime(self.data_info.iloc[:, 6], format="%Y%m%dT%H%M%S")
        self.pastdate = (self.pastdate - pd.to_datetime("20150623", format="%Y%m%d")).dt.days
        self.presentdate = (self.presentdate - pd.to_datetime("20150623", format="%Y%m%d")).dt.days
        self.time = np.concatenate((self.pastdate, self.presentdate))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.data_len = len(self.low)
#         self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)
        # self.crop = tt.RandomCrop(resolution)
#         print(self.naip.shape, self.sentinel.shape, self.time.shape)

    def __getitem__(self, index):
        # Get image name from the pandas df
        high = self.high[index]
        path = high.split("/")[-1].split(".")[0]+"_"+str(index)
        low = self.low[index]
        t = self.time[index]
        if index >= self.house_count:
            high2 = self.high[index-self.house_count]
        else:
            high2 = self.highpresent[index]
        if index == 0:
            print(high, high2)
        # Open image
        high = Image.open(high).convert('RGB')
        high2 = Image.open(high2).convert('RGB')
        low = Image.open(low).convert('RGB')
        # Transform the image
        high = self.transform(high)
        high2 = self.transform(high2)
        low = self.enc_transform(low)
        coords = tt.convert_to_coord_uneven_t(1, self.resolution, self.resolution, t, integer_values=self.integer_values)
#         print(coords)
        high = torch.cat([high, coords], 0)

        return (high, low, high2, path)

    def __len__(self):
        return self.data_len

class FMoWSentinelPatch(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution, crop_size, integer_values):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.integer_values = integer_values
        self.resolution = resolution
        self.crop_size = crop_size
        self.crop = tt.RandomCropDim3(crop_size)
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.highpast = np.asarray(self.data_info.iloc[:, 1])
        self.highpresent = np.asarray(self.data_info.iloc[:, 2])
        self.high = np.concatenate((self.highpast, self.highpresent))
        self.lowpast = np.asarray(self.data_info.iloc[:, 3])
        self.lowpresent = np.asarray(self.data_info.iloc[:, 4])
        self.low = np.concatenate((self.lowpast, self.lowpresent))
        #process date
        self.pastdate = pd.to_datetime(self.data_info.iloc[:, 5], format="%Y%m%dT%H%M%S")
        self.presentdate = pd.to_datetime(self.data_info.iloc[:, 6], format="%Y%m%dT%H%M%S")
        self.pastdate = (self.pastdate - pd.to_datetime("20150623", format="%Y%m%d")).dt.days
        self.presentdate = (self.presentdate - pd.to_datetime("20150623", format="%Y%m%d")).dt.days
        self.time = np.concatenate((self.pastdate, self.presentdate))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.data_len = len(self.low)
#         self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)
        # self.crop = tt.RandomCrop(resolution)
#         print(self.naip.shape, self.sentinel.shape, self.time.shape)

#     def get_sinusoid_encoding_table(positions, d_hid, T=1000):
#         ''' Sinusoid position encoding table
#         positions: int or list of integer, if int range(positions)'''

#         if isinstance(positions, int):
#             positions = list(range(positions))

#         def cal_angle(position, hid_idx):
#             return position / np.power(T, 2 * (hid_idx // 2) / d_hid)

#         def get_posi_angle_vec(position):
#             return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

#         sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in positions])

#         sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
#         sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

#         if torch.cuda.is_available():
#             return torch.FloatTensor(sinusoid_table).cuda()
#         else:
#             return torch.FloatTensor(sinusoid_table)

    def __getitem__(self, index):
        # Get image name from the pandas df
        high = self.high[index]
        low = self.low[index]
        t = self.time[index]
        if index >= self.house_count:
            high2 = self.high[index-self.house_count]
        else:
            high2 = self.highpresent[index]
#         if index == 0:
#             print(high, high2)
        # Open image
        high = Image.open(high).convert('RGB')
        high2 = Image.open(high2).convert('RGB')
        low = Image.open(low).convert('RGB')
        # Transform the image
        high = self.transform(high)
        high2 = self.transform(high2)
        low = self.enc_transform(low)
        coords = tt.convert_to_coord_uneven_t(1, self.resolution, self.resolution, t, integer_values=self.integer_values)
#         print(coords)
        high = torch.cat([high, coords], 0)

        high, h_start, w_start = self.crop(high)
        high2 = tt.patch_crop_dim3(high2, h_start, w_start, self.crop_size)
        low = tt.patch_crop_dim3(low, h_start, w_start, self.crop_size)

        return (high, low, high2, h_start, w_start)

    def __len__(self):
        return self.data_len

class FSAllPatch(Dataset):
    def __init__(self, csv_path, transform, enc_transform, resolution, crop_size, integer_values):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        self.integer_values = integer_values
        self.resolution = resolution
        self.crop_size = crop_size
        self.n = resolution // crop_size
        # Transforms
        self.transform = transform
        self.enc_transform = enc_transform
        # Read the csv file
        self.data_info = pd.read_csv(csv_path, header=0)
        # column 1-4 contain the image paths
        self.highpast = np.asarray(self.data_info.iloc[:, 1])
        self.highpresent = np.asarray(self.data_info.iloc[:, 2])
        self.high = np.concatenate((self.highpast, self.highpresent))
        self.lowpast = np.asarray(self.data_info.iloc[:, 3])
        self.lowpresent = np.asarray(self.data_info.iloc[:, 4])
        self.low = np.concatenate((self.lowpast, self.lowpresent))
        #process date
        self.pastdate = pd.to_datetime(self.data_info.iloc[:, 5], format="%Y%m%dT%H%M%S")
        self.presentdate = pd.to_datetime(self.data_info.iloc[:, 6], format="%Y%m%dT%H%M%S")
        self.pastdate = (self.pastdate - pd.to_datetime("20150623", format="%Y%m%d")).dt.days
        self.presentdate = (self.presentdate - pd.to_datetime("20150623", format="%Y%m%d")).dt.days
        self.time = np.concatenate((self.pastdate, self.presentdate))
        # Calculate len
        data_len = len(self.data_info.index)
        self.house_count = data_len
        self.data_len = len(self.low)
#         self.coords = tt.convert_to_coord_with_t(1, resolution, resolution, [0,1], integer_values=self.integer_values)
        # self.crop = tt.RandomCrop(resolution)
#         print(self.naip.shape, self.sentinel.shape, self.time.shape)

    def __getitem__(self, index):
        # Get image name from the pandas df
        data = {}
        high = self.high[index]
        path = high.split("/")[-1].split(".")[0]+"_"+str(index)
        low = self.low[index]
        t = self.time[index]
        if index >= self.house_count:
            high2 = self.high[index-self.house_count]
        else:
            high2 = self.highpresent[index]
#         if index == 0:
#             print(high, high2)
        # Open image
        high = Image.open(high).convert('RGB')
        high2 = Image.open(high2).convert('RGB')
        low = Image.open(low).convert('RGB')
        # Transform the image
        high = self.transform(high)
        high2 = self.transform(high2)
        low = self.enc_transform(low)
        coords = tt.convert_to_coord_uneven_t(1, self.resolution, self.resolution, t, integer_values=self.integer_values)
#         print(coords)
        high = torch.cat([high, coords], 0)

        for i in range(self.n):
            for j in range(self.n):
                high_ij = tt.patch_crop_dim3(high, i*self.crop_size, j*self.crop_size, self.crop_size)
                high2_ij = tt.patch_crop_dim3(high2, i*self.crop_size, j*self.crop_size, self.crop_size)
                low_ij = tt.patch_crop_dim3(low, i*self.crop_size, j*self.crop_size, self.crop_size)
                data[(i,j)] = (high_ij, low_ij, high2_ij, i*self.crop_size, j*self.crop_size)
        half_size = self.crop_size//2
        for i in range(self.n):
            for j in range(self.n-1):
                high_ij = tt.patch_crop_dim3(high, i*self.crop_size, j*self.crop_size+half_size, self.crop_size)
                high2_ij = tt.patch_crop_dim3(high2, i*self.crop_size, j*self.crop_size+half_size, self.crop_size)
                low_ij = tt.patch_crop_dim3(low, i*self.crop_size, j*self.crop_size+half_size, self.crop_size)
                data[(i,j+0.5)] = (high_ij, low_ij, high2_ij, i*self.crop_size, j*self.crop_size)
        for j in range(self.n):
            for i in range(self.n-1):
                high_ij = tt.patch_crop_dim3(high, i*self.crop_size+half_size, j*self.crop_size, self.crop_size)
                high2_ij = tt.patch_crop_dim3(high2, i*self.crop_size+half_size, j*self.crop_size, self.crop_size)
                low_ij = tt.patch_crop_dim3(low, i*self.crop_size+half_size, j*self.crop_size, self.crop_size)
                data[(i+0.5,j)] = (high_ij, low_ij, high2_ij, i*self.crop_size, j*self.crop_size)

        return data, path

    def __len__(self):
        return self.data_len
