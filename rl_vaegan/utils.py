"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import math
import os

import numpy as np
import torch
import torch.nn.init as init
# from torch.utils.serialization import load_lua
import torchfile
import torchvision.utils as vutils
import yaml
from torch.autograd import Variable
from torch.optim import lr_scheduler
from torchvision import transforms

from rl_vaegan.networks import Vgg16


def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def __write_images(image_outputs, display_image_num, file_name):
    image_outputs = [images.expand(-1, 3, -1, -1) for images in image_outputs
                     ]  # expand gray-scale images to 3 channels
    image_tensor = torch.cat(
        [images[:display_image_num] for images in image_outputs], 0)
    image_grid = vutils.make_grid(image_tensor.data,
                                  nrow=display_image_num,
                                  padding=0,
                                  normalize=True)
    vutils.save_image(image_grid, file_name, nrow=1)


def write_2images(image_outputs, display_image_num, image_directory, postfix):
    n = len(image_outputs)
    __write_images(image_outputs[0:n // 2], display_image_num,
                   '%s/gen_a2b_%s.jpg' % (image_directory, postfix))
    __write_images(image_outputs[n // 2:n], display_image_num,
                   '%s/gen_b2a_%s.jpg' % (image_directory, postfix))


def prepare_sub_folder(output_directory):
    image_directory = os.path.join(output_directory, 'results')
    if not os.path.exists(image_directory):
        print("Creating directory: {}".format(image_directory))
        os.makedirs(image_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, image_directory


def write_loss(iterations, trainer, train_writer):
    members = [
        attr for attr in dir(trainer)
        if not callable(getattr(trainer, attr)) and not attr.startswith("__")
        and ('loss' in attr or 'grad' in attr or 'nwd' in attr)
    ]
    for m in members:
        train_writer.add_scalar(m, getattr(trainer, m), iterations + 1)


# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [
        os.path.join(dirname, f) for f in os.listdir(dirname)
        if os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f
    ]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name


def load_vgg16(model_dir):
    """ Use the model from https://github.com/abhiskk/fast-neural-style/blob/master/neural_style/utils.py """
    if not os.path.exists(model_dir):
        # os.mkdir(model_dir)
        os.makedirs(model_dir)
    print(model_dir)
    if not os.path.exists(os.path.join(model_dir, 'vgg16.weight')):
        # if not os.path.exists(os.path.join(model_dir, 'vgg16.t7')):
        #     os.system('wget https://www.dropbox.com/s/76l3rt4kyi3s8x7/vgg16.t7?dl=1 -O ' + os.path.join(model_dir, 'vgg16.t7'))
        vgglua = torchfile.load(os.path.join(model_dir, 'vgg16.t7'))
        vgg = Vgg16()
        for (src, dst) in zip(vgglua.parameters()[0], vgg.parameters()):
            dst.data[:] = src
        torch.save(vgg.state_dict(), os.path.join(model_dir, 'vgg16.weight'))
    vgg = Vgg16()
    vgg.load_state_dict(torch.load(os.path.join(model_dir, 'vgg16.weight')))
    return vgg


def vgg_preprocess(batch):  #torch.Size([1, 3, 80, 80])
    tensortype = type(batch.data)
    (r, g, b) = torch.chunk(batch, 3, dim=1)
    batch = torch.cat((b, g, r), dim=1)  # convert RGB to BGR
    batch = (batch + 1) * 255 * 0.5  # [-1, 1] -> [0, 255]
    mean = tensortype(batch.data.size()).cuda()
    mean[:, 0, :, :] = 103.939
    mean[:, 1, :, :] = 116.779
    mean[:, 2, :, :] = 123.680
    batch = batch.sub(Variable(mean))  # subtract mean
    return batch


def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters[
            'lr_policy'] == 'constant':
        scheduler = None  # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'],
                                        last_epoch=iterations)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented',
            hyperparameters['lr_policy'])
    return scheduler


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0
                or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)

    return init_fun
