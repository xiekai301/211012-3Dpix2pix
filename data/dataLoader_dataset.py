import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import torchio as tio
import numpy as np
import nibabel as nib
import torch.nn.functional as nnf
import torch
import matplotlib.pyplot as plt
import SimpleITK as sitk


def save3Dimage(img3d, img_shape, path):
        img3d = torch.squeeze(img3d, 0)
        img_shape = img3d.shape

        plt.subplot(2, 2, 1)
        plt.imshow(img3d[:, :, img_shape[2]//2], cmap="gray")
        # a1.set_aspect(ax_aspect)

        plt.subplot(2, 2, 2)
        plt.imshow(img3d[:, img_shape[1]//2, :], cmap="gray")
        # a2.set_aspect(sag_aspect)

        plt.subplot(2, 2, 3)
        plt.imshow(img3d[img_shape[0]//2, :, :].T, cmap="gray")
        # a3.set_aspect(cor_aspect)

        plt.savefig(path)

def save3Dimage_numpy(img3d, img_shape, path):

        plt.subplot(2, 2, 1)
        plt.imshow(img3d[:, :, img_shape[2]//2], cmap="gray")
        # a1.set_aspect(ax_aspect)

        plt.subplot(2, 2, 2)
        plt.imshow(img3d[:, img_shape[1]//2, :], cmap="gray")
        # a2.set_aspect(sag_aspect)

        plt.subplot(2, 2, 3)
        plt.imshow(img3d[img_shape[0]//2, :, :].T, cmap="gray")
        # a3.set_aspect(cor_aspect)

        plt.savefig(path)
        
def read_nii(nii_path):
    img = sitk.ReadImage(nii_path)
    img = sitk.GetArrayFromImage(img)
    truncted_num = img.shape[0] % 16
    img = img[:img.shape[0] - truncted_num, :, :]
    img[img < 0] = 0
    img[img > 3000] = 3000
    img = 2 * img / 3000 - 1  # nii follewed by matlab
    return img

def read_edge_nii(nii_edge_path):
    edge = sitk.ReadImage(nii_edge_path)
    edge = sitk.GetArrayFromImage(edge)
    truncted_num = edge.shape[0] % 16
    edge = edge[:edge.shape[0] - truncted_num, :, :]
    edge[edge < 1] = -1

    return edge

def circlemask_cropped(input_shape):
    # D, H, W, _ = x.shape
    D, H, W= input_shape
    x, y = np.ogrid[:H, :W]
    cx, cy = H / 2, W / 2
    radius = int(np.random.uniform(0.75, 0.75) * H / 2)
    r2 = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    circmask = r2 > radius * radius
    mask = np.expand_dims(circmask, axis=[0,1,-1]).repeat([D, ], axis=1)
    return mask


class DataLoaderDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.
    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        self.dir_ct = os.path.join(opt.dataroot, opt.phase)  # create a path '/path/to/data/ct'
        # self.dir_mr = os.path.join(opt.dataroot, opt.phase + 'mr')  # create a path '/path/to/data/mr'
        # self.dir_ct_label = os.path.join(opt.dataroot, 'trainct_lables')  # create a path '/path/to/data/ct'
        # self.dir_mr_label = os.path.join(opt.dataroot, 'trainmr_lables')  # create a path '/path/to/data/mr'
        #
        self.nii_paths = sorted(make_dataset(self.dir_ct, opt.max_dataset_size))   # load images from '/path/to/data/ct'
        # self.mr_paths = sorted(make_dataset(self.dir_mr, opt.max_dataset_size))    # load images from '/path/to/data/mr'
        # self.nii_paths_label = sorted(make_dataset(self.dir_ct_label, opt.max_dataset_size))   # load images from '/path/to/data/ct'
        # self.mr_paths_label = sorted(make_dataset(self.dir_mr_label, opt.max_dataset_size))    # load images from '/path/to/data/mr'
        #
        self.ct_size = len(self.nii_paths)  # get the size of dataset ct
        # self.mr_size = len(self.mr_paths)  # get the size of dataset mr
        # self.ct_size_label = len(self.nii_paths_label)  # get the size of dataset ct
        # self.mr_size_label = len(self.mr_paths_label)  # get the size of dataset mr
        mrtoct = self.opt.direction == 'mrtoct'
        input_nc = self.opt.output_nc if mrtoct else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if mrtoct else self.opt.output_nc      # get the number of channels of output image
        # self.transform_ct = get_transform(self.opt)
        # self.transform_mr = get_transform(self.opt)


    def __getitem__(self, index):
        """Return a data point and its metadata information.
        Parameters:
            index (int)      -- a random integer for data indexing
        Returns a dictionary that contains ct, mr, nii_paths and mr_paths
            ct (tensor)       -- an image in the input domain
            mr (tensor)       -- its corresponding image in the target domain
            nii_paths (str)    -- image paths
            mr_paths (str)    -- image paths
        """
        nii_path = self.nii_paths[index % self.ct_size]  # make sure index is within then range
        edge_path = nii_path.replace('.nii', '_edge.nii.gz')
        img = read_nii(nii_path)
        edge = read_edge_nii(edge_path)
        # mask = circlemask_cropped(img.shape)
        # input = img.copy()
        # input[mask] = -1

        # ------------------------------------------------
        # return {'ct': ct, 'mr': mr, 'nii_paths': nii_path, 'mr_paths': mr_path, 'ct_label': ct_label, 'mr_label': mr_label}
        # return {'A': ct, 'B': mr, 'A_paths': nii_path, 'B_paths': mr_path}
        return {'img': img, 'edge': edge, 'A_paths': nii_path}

    def __len__(self):
        """Return the total number of images in the dataset.
        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        # return max(self.ct_size, self.mr_size)
        return self.ct_size

