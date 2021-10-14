"""General-purpose test script for image-to-image translation.

Once you have trained your model with train.py, you can use this script to test the model.
It will load a saved model from '--checkpoints_dir' and save the results to '--results_dir'.

It first creates model and dataset given the option. It will hard-code some parameters.
It then runs inference for '--num_test' images and save results to an HTML file.

Example (You need to train models first or download pre-trained models from our website):
    Test a CycleGAN model (both sides):
        python test.py --dataroot ./datasets/maps --name maps_cyclegan --model cycle_gan

    Test a CycleGAN model (one side only):
        python test.py --dataroot datasets/horse2zebra/testA --name horse2zebra_pretrained --model test --no_dropout

    The option '--model test' is used for generating CycleGAN results only for one side.
    This option will automatically set '--dataset_mode single', which only loads the images from one set.
    On the contrary, using '--model cycle_gan' requires loading and generating results in both directions,
    which is sometimes unnecessary. The results will be saved at ./results/.
    Use '--results_dir <directory_path_to_save_result>' to specify the results directory.

    Test a pix2pix model:
        python test.py --dataroot ./datasets/facades --name facades_pix2pix --model pix2pix --direction BtoA

See options/base_options.py and options/test_options.py for more test options.
See training and test tips at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/tips.md
See frequently asked questions at: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/docs/qa.md
"""
import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import torch
import numpy as np
import SimpleITK as sitk
from util.extraction import extract_patches, reconstruct_volume, circlemask_cropped
input_shape = [32,256,256,1]
IMG_STRIDE = 16


if __name__ == '__main__':
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers
    # create a website
    # web_dir = os.path.join(opt.results_dir, opt.name, '{}_{}'.format(opt.phase, opt.epoch))  # define the website directory
    # if opt.load_iter > 0:  # load_iter is 0 by default
    #     web_dir = '{:s}_iter{:d}'.format(web_dir, opt.load_iter)
    # print('creating web directory', web_dir)
    # webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.epoch))
    # test with eval mode. This only affects layers like batchnorm and dropout.
    # For [pix2pix]: we use batchnorm and dropout in the original pix2pix. You can experiment it with and without eval() mode.
    # For [CycleGAN]: It should not affect CycleGAN as CycleGAN uses instancenorm without dropout.
    if opt.eval:
        model.eval()
    for i, data in enumerate(dataset):

        img = np.array(data['img']).squeeze()
        edge = np.array(data['edge']).squeeze()
        patches = extract_patches(img=img, patch_shape=input_shape[:3], extraction_step=[IMG_STRIDE, 1, 1])
        edge_patches = extract_patches(img=edge, patch_shape=input_shape[:3], extraction_step=[IMG_STRIDE, 1, 1])
        for l in range(patches.shape[0] - opt.batch_size + 1):
            # for l in range(1):
            patch = patches[l:l + opt.batch_size, :]  # patch排序
            edge_patch = edge_patches[l:l + opt.batch_size, :]
            mask = circlemask_cropped(input_shape)
            mask = np.expand_dims(mask, axis=[0, 1]).repeat(opt.batch_size, axis=0)
            edge = np.expand_dims(edge_patch, axis=1)
            patch = np.expand_dims(patch, axis=1)
            target = patch
            input_image = target.copy()
            input_image[mask] = -1
            input_image = torch.from_numpy(input_image.astype('float32'))
            input_edge = torch.from_numpy(edge.astype('float32'))
            input_mask = torch.from_numpy(mask.astype('float32'))
            target = torch.from_numpy(target.astype('float32'))
            data_patch = {'A': input_image, 'edge': input_edge, 'mask': input_mask,
                          'B': target, 'A_paths': data['A_paths'], 'B_paths': data['A_paths']}

            model.set_input(data_patch)  # unpack data from data loader
            model.test()           # run inference
            visuals = model.get_current_visuals()  # get image results
            fakeB = visuals['fake_B'].cpu().numpy()
            patches[l, :] = fakeB.numpy()[0, 0, ...]

        recon_img = reconstruct_volume(patches=patches, expected_shape=img.shape, extraction_step=(16, 1, 1))
        recon_img = (recon_img * 0.5) + 0.5
        recon_img = recon_img * 3000
        volout = sitk.GetImageFromArray(recon_img.astype(np.int16))
        img_path = model.get_image_paths()
        sitk.WriteImage(volout, 'output/' + img_path.split('/')[-1])

        if i % 5 == 0:  # save images to an HTML file
            print('processing (%04d)-th image... %s' % (i, img_path))
