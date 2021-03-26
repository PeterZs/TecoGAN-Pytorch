import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


from .paired_lmdb_dataset import PairedLMDBDataset
from .unpaired_lmdb_dataset import UnpairedLMDBDataset
from .paired_folder_dataset import PairedFolderDataset
import numpy as np
import random

def create_dataloader(opt, dataset_idx='train'):
    # setup params
    data_opt = opt['dataset'].get(dataset_idx)
    degradation_type = opt['dataset']['degradation']['type']

    # -------------- loader for training -------------- #
    if dataset_idx == 'train':
        # check dataset
        assert data_opt['name'] in ('VimeoTecoGAN', 'VimeoTecoGAN-sub'), \
            'Unknown Dataset: {}'.format(data_opt['name'])

        if degradation_type == 'BI':
            # create dataset
            dataset = PairedLMDBDataset(
                data_opt,
                scale=opt['scale'],
                tempo_extent=opt['train']['tempo_extent'],
                moving_first_frame=opt['train'].get('moving_first_frame', False),
                moving_factor=opt['train'].get('moving_factor', 1.0))

        elif degradation_type == 'BD':
            # enlarge crop size to incorporate border size
            sigma = opt['dataset']['degradation']['sigma']
            enlarged_crop_size = data_opt['crop_size'] + 2 * int(sigma * 3.0)

            # create dataset
            dataset = UnpairedLMDBDataset(
                data_opt,
                crop_size=enlarged_crop_size,  # override
                tempo_extent=opt['train']['tempo_extent'],
                moving_first_frame=opt['train'].get('moving_first_frame', False),
                moving_factor=opt['train'].get('moving_factor', 1.0))

        else:
            raise ValueError('Unrecognized utils type: {}'.format(
                degradation_type))

        # create data loader
        loader = DataLoader(
            dataset=dataset,
            batch_size=data_opt['batch_size'],
            shuffle=True,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    # -------------- loader for testing -------------- #
    elif dataset_idx.startswith('test'):
        # create data loader
        loader = DataLoader(
            dataset=PairedFolderDataset(
                data_opt,
                scale=opt['scale']),
            batch_size=1,
            shuffle=False,
            num_workers=data_opt['num_workers'],
            pin_memory=data_opt['pin_memory'])

    else:
        raise ValueError('Unrecognized dataset index: {}'.format(dataset_idx))

    return loader




def prepare_data(opt, data, kernel):
    """ prepare gt, lr data for training

        for BD utils, generate lr data and remove border of gt data
        for BI utils, return data directly

    """

    device = torch.device(opt['device'])
    degradation_type = opt['dataset']['degradation']['type']

    if degradation_type == 'BI':
        gt_data, lr_data = data['gt'].to(device), data['lr'].to(device)

    elif degradation_type == 'BD':
        # setup params
        scale = opt['scale']
        sigma = opt['dataset']['degradation'].get('sigma', 1.5)
        border_size = int(sigma * 3.0)

        gt_with_border = data['gt'].to(device)
        n, t, c, gt_h, gt_w = gt_with_border.size()
        lr_h = (gt_h - 2 * border_size) // scale
        lr_w = (gt_w - 2 * border_size) // scale

        # generate lr data
        gt_with_border = gt_with_border.view(n * t, c, gt_h, gt_w)
        lr_data = F.conv2d(
            gt_with_border, kernel, stride=scale, bias=None, padding=0)
        lr_data = lr_data.view(n, t, c, lr_h, lr_w)

        # remove gt border
        gt_data = gt_with_border[
            ...,
            border_size: border_size + scale * lr_h,
            border_size: border_size + scale * lr_w
        ]
        gt_data = gt_data.view(n, t, c, scale * lr_h, scale * lr_w)

    else:
        raise ValueError('Unrecognized utils type: {}'.format(
            degradation_type))

    return { 'gt': gt_data, 'lr': lr_data }


def prepare_data_face_degradation(opt, data, kernel, Distort):
    """ prepare gt, lr data for training

        for BD utils, generate lr data and remove border of gt data
        for BI utils, return data directly

    """

    device = torch.device(opt['device'])
    degradation_type = opt['dataset']['degradation']['type']

    if degradation_type == 'BI':
        gt_data, lr_data = data['gt'].to(device), data['lr'].to(device)

    elif degradation_type == 'BD':
        # setup params
        scale = opt['scale']
        sigma = opt['dataset']['degradation'].get('sigma', 1.5)
        border_size = int(sigma * 3.0)

        gt_with_border = data['gt'].to(device)
        n, t, c, gt_h, gt_w = gt_with_border.size()
        lr_h = (gt_h - 2 * border_size) // scale
        lr_w = (gt_w - 2 * border_size) // scale

        # generate lr data
        gt_with_border = gt_with_border.view(n * t, c, gt_h, gt_w)

        # original
        # lr_data = F.conv2d(
        #     gt_with_border, kernel, stride=scale, bias=None, padding=0)

        # face restoration utils
        [n,t,c,h,w] = list(data['gt'].shape)

        Size  = (h,w)
        lr_data = []
        for i in range(n):
            seed_ = random.randint(1,1000)
            for j in range(t):
                gt_image = torch.squeeze(data['gt'][i,j,:,:,:],0)
                gt_image = gt_image.permute([1,2,0]).numpy()
                gt_image = np.uint8(gt_image)
                gt_raw, lr_data_, gain = Distort.Distort_random_v5(gt_image, Size,seed_)
                lr_data_ = torch.from_numpy(lr_data_)
                lr_data_ = lr_data_.permute([2,0,1])
                lr_data_ = lr_data_.unsqueeze(0)
                lr_data.append(lr_data_)
        lr_data = torch.cat(lr_data, 0)
        lr_data = lr_data.unsqueeze(0).float().to(device)
        lr_data = lr_data[
            ...,
            border_size: border_size + scale * lr_h,
            border_size: border_size + scale * lr_w
        ]
        # remove gt border
        gt_data = gt_with_border[
            ...,
            border_size: border_size + scale * lr_h,
            border_size: border_size + scale * lr_w
        ]
        gt_data = gt_data.view(n, t, c, scale * lr_h, scale * lr_w)

    else:
        raise ValueError('Unrecognized utils type: {}'.format(
            degradation_type))

    return { 'gt': gt_data, 'lr': lr_data }

