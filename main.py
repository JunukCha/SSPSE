import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"

from trainer import Trainer
from model.encoder import get_encoder
from model.texture_net import TextureNet
from lib.models.spin import get_pretrained_hmr
from model.discriminator import get_discriminator
from utils.part_utils import PartRenderer
from utils.render_utils import IMGRenderer

from dataset.base_dataset import BaseDataset
from dataset.mixed_dataset import MixedDataset
from utils.trainer_utils import get_HHMMSS_from_second

import torch
import torch.nn as nn
import torch.optim as optim

import time
import argparse


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def main(args):
    device = torch.device("cuda:{}".format(args.GPU_ORDER) if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("Current GPU Order: {}".format(torch.cuda.current_device()))
    # model
    model_name = "HMR"
    HMR = get_pretrained_hmr(device=device)

    ### Context Encoder (part of Decoder)
    context_encoder_net = get_encoder("ContextEncoderNet")

    ### Context Encoder (discriminator)
    discriminator = get_discriminator("Discriminator")

    ### JigsawPuzzle (part of classification)
    jigsaw_puzzle_net = get_encoder("JigsawPuzzleNet")

    ### Rotation (part of classification)
    rotation_net = get_encoder("RotationNet")
    
    ### TextureNet
    texture_net = TextureNet(args.tex_size)

    ### RendererNet
    img_renderer = IMGRenderer()
    seg_renderer = PartRenderer()

    ### Discriminator for texture
    texture_discriminator = get_discriminator("Discriminator")
    
    ### Context Encoder Intialized and optimizer setting ###
    if args.train == 1:
        context_encoder_net.apply(weights_init)
    
    context_encoder_optimizer = optim.Adam(context_encoder_net.parameters(), lr=args.lr_context)
    
    ### Discriminator for CE Intialized and optimizer setting ###
    if args.train == 1:
        discriminator.apply(weights_init)

    discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=args.lr_disc)

    ### Jigsaw Puzzle Net Intialized and optimizer setting ###
    if args.train == 1:
        jigsaw_puzzle_net.apply(weights_init)

    jigsaw_puzzle_optimizer = optim.Adam(jigsaw_puzzle_net.parameters(), lr=args.lr_jigsaw)

    ### Rotation Net Intialized and optimizer setting ###
    if args.train == 1:
        rotation_net.apply(weights_init)

    rotation_optimizer = optim.Adam(rotation_net.parameters(), lr=args.lr_rot)

    ### Texture_net Initalized and optimizer setting ###
    if args.train == 1:
        texture_net.apply(weights_init)

    texture_net_optimizer = optim.Adam(texture_net.parameters(), lr=args.lr_texture_net)

    ### Discriminator for texture Intialized and optimizer setting ###
    texture_discriminator_optimizer = optim.Adam(texture_discriminator.parameters(), lr=args.lr_texture_discriminator)

    ### HMR Initialized for only evaluation ###
    if args.train == 0 and args.checkpoint:
        save_HMR_pth_path = args.checkpoint
        HMR_checkpoint = torch.load(save_HMR_pth_path, map_location=device)
        HMR.load_state_dict(HMR_checkpoint['state_dict'])

    HMR_optimizer_all = optim.Adam(HMR.parameters(), lr=args.lr_hmr_all)

    ### Optimizer scheduler ###
    HMR_scheduler_all = optim.lr_scheduler.ExponentialLR(HMR_optimizer_all, gamma=args.gamma_joints)

    ### Loss function ###
    loss_fn_BCE = nn.BCELoss()
    loss_fn_CE = nn.CrossEntropyLoss()
    loss_fn_MSE = nn.MSELoss()
    loss_fn_keypoints = nn.MSELoss(reduction="none")
    loss_fn_mask = nn.MSELoss()

    
 
    if args.train == 0: # Test
        train_datasets = None
        train_dataloader = None

        test_datasets = BaseDataset(None, args.test_dataset, is_train=False)
        
        test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=args.test_batch_size, num_workers=args.num_workers)
        test_dataloader_h36m = None
        test_dataloader_3dpw = None
        test_dataloader_lsp = None

    else: # Train
        train_datasets = MixedDataset(args, ignore_3d=args.ignore_3d, is_train=True)
        train_dataloader = torch.utils.data.DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        test_datasets_h36m = BaseDataset(args, "h36m-p2", is_train=False)
        test_datasets_3dpw = BaseDataset(args, "3dpw", is_train=False)
        test_datasets_lsp = BaseDataset(args, "lsp", is_train=False)

        test_dataloader = None
        test_dataloader_h36m = torch.utils.data.DataLoader(test_datasets_h36m, batch_size=args.test_batch_size, num_workers=args.num_workers)
        test_dataloader_3dpw = torch.utils.data.DataLoader(test_datasets_3dpw, batch_size=args.test_batch_size, num_workers=args.num_workers)
        test_dataloader_lsp = torch.utils.data.DataLoader(test_datasets_lsp, batch_size=args.test_batch_size, num_workers=args.num_workers)
        

    # GPU setting
    device = torch.device("cuda:{}".format(args.GPU_ORDER) if torch.cuda.is_available() else "cpu")
    HMR = HMR.to(device)
    context_encoder_net = context_encoder_net.to(device)
    jigsaw_puzzle_net = jigsaw_puzzle_net.to(device)
    rotation_net = rotation_net.to(device)
    discriminator = discriminator.to(device)
    texture_net = texture_net.to(device)
    texture_discriminator = texture_discriminator.to(device)

    Trainer(
        model_name=model_name,
        HMR=HMR,
        context_encoder_net=context_encoder_net,
        discriminator=discriminator,
        jigsaw_puzzle_net=jigsaw_puzzle_net,
        rotation_net=rotation_net,
        texture_net=texture_net,
        img_renderer=img_renderer,
        seg_renderer=seg_renderer,
        texture_discriminator=texture_discriminator,
        
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        test_dataloader_h36m=test_dataloader_h36m,
        test_dataloader_3dpw=test_dataloader_3dpw,
        test_dataloader_lsp=test_dataloader_lsp,

        loss_fn_BCE=loss_fn_BCE,
        loss_fn_CE=loss_fn_CE,
        loss_fn_MSE=loss_fn_MSE,
        loss_fn_mask=loss_fn_mask,
        loss_fn_keypoints=loss_fn_keypoints,

        HMR_optimizer_all=HMR_optimizer_all,
        HMR_scheduler_all=HMR_scheduler_all,

        discriminator_optimizer=discriminator_optimizer,
        context_encoder_optimizer=context_encoder_optimizer,
        jigsaw_puzzle_optimizer=jigsaw_puzzle_optimizer,
        rotation_optimizer=rotation_optimizer,
        texture_net_optimizer=texture_net_optimizer,
        texture_discriminator_optimizer=texture_discriminator_optimizer,

        device=device,
        num_epoch=args.nEpoch,
        args=args
    ).fit()


if __name__ == '__main__':
    start = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument('--nEpoch', type=int, default=20)
    parser.add_argument('--first_stage_nEpoch', type=int, default=10)

    parser.add_argument('--lr_hmr_all', type=float, default=5e-6)
    parser.add_argument('--lr_hmr_resnet', type=float, default=5e-6)
    parser.add_argument('--lr_context', type=float, default=0.0001)
    parser.add_argument('--lr_disc', type=float, default=0.0001)
    parser.add_argument('--lr_jigsaw', type=float, default=0.01)
    parser.add_argument('--lr_rot', type=float, default=0.0005)
    parser.add_argument('--lr_texture_net', type=float, default=0.01)
    parser.add_argument('--lr_texture_discriminator', type=float, default=0.0001)

    parser.add_argument('--gamma_joints', type=float, default=0.99)
    parser.add_argument('--gamma_3task', type=float, default=0.99)

    parser.add_argument('--noise_factor', type=float, default=0.4)
    parser.add_argument('--scale_factor', type=float, default=0.25)
    parser.add_argument('--rot_factor', type=float, default=30)
    parser.add_argument('--ignore_3d', default=False, action='store_true')
    parser.add_argument('--self_supervised', default=False, action='store_true')
    
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--test_batch_size', type=int, default=600)

    parser.add_argument('--GPU_ORDER', type=int, default=0,
                        help='Order of GPU used')
    
    parser.add_argument('--train', type=int, default=0,
                        help='0 => test\
                              1 => train \
                              ')

    parser.add_argument('--freq_print', type=int, default=1000)
    parser.add_argument('--freq_print_test', type=int, default=20)

    parser.add_argument('--freq_eval', type=int, default=1)

    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--tex_size', type=int, default=2)

    parser.add_argument('--test_dataset', type=str, 
                        default="3dpw", choices=["3dpw", "h36m-p2", "lsp"])

    parser.add_argument('--output_dir', type=str)

    parser.add_argument('--ce_weight', type=float, default=1.0)
    parser.add_argument('--jp_weight', type=float, default=1.0)
    parser.add_argument('--rot_weight', type=float, default=1.0)
    parser.add_argument('--total_weight', type=float, default=1.0)

    parser.add_argument('--rendering_weight', type=float, default=1.0)
    parser.add_argument('--seg_weight', type=float, default=1.0)
    parser.add_argument('--texture_total_weight', type=float, default=1.0)
    parser.add_argument('--gan_loss_weight', type=float, default=0.001)

    parser.add_argument('--first_eval', action='store_true')
    
    parser.add_argument('--save_pth_all_epoch', action='store_true')

    parser.add_argument('--checkpoint', type=str)

    args = parser.parse_args()
    
    main(args)

    duration = time.time() - start
    duration = get_HHMMSS_from_second(duration)
    print("Time: {}".format(duration))