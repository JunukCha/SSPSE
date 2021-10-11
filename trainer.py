import os, datetime, time
import torch
import torch.optim as optim
import numpy as np
import math
import cv2
import tqdm
import config

import constants
from utils.trainer_utils import (
    AverageMeter,
    get_HHMMSS_from_second, 
    save_checkpoint, 
    save_all_img,
    save_joints3d_img,
    save_mesh,
    save_templates_info,

    train_only_3task_network,
    train_hmr_using_3task,
    train_hmr_using_joints,
    train_texture_net,
    train_hmr_using_adv_loss,
)

from utils.imutils import uncrop

from lib.utils.eval_utils import (
    batch_compute_similarity_transform_torch,
)
from lib.utils.geometry import batch_rodrigues
from lib.models.smpl import SMPL, SMPL_MODEL_DIR, H36M_TO_J14, SMPL_MEAN_PARAMS

# import soft_renderer as sr
# from soft_renderer.mesh import Mesh
# from soft_renderer.renderer import SoftRenderer
# import soft_renderer.cuda.load_textures as load_textures_cuda
from lib.models.smpl import get_smpl_faces


class Trainer():
    def __init__(
        self,
        model_name,

        HMR,
        context_encoder_net,
        discriminator,
        jigsaw_puzzle_net,
        rotation_net,
        texture_net,
        img_renderer,
        seg_renderer,
        texture_discriminator,

        train_dataloader,
        test_dataloader,
        test_dataloader_h36m,
        test_dataloader_3dpw,
        test_dataloader_lsp,

        loss_fn_BCE,
        loss_fn_CE,
        loss_fn_MSE,
        loss_fn_keypoints,
        loss_fn_mask,

        HMR_optimizer_all,
        HMR_scheduler_all,
        discriminator_optimizer,
        context_encoder_optimizer,
        jigsaw_puzzle_optimizer,
        rotation_optimizer,
        texture_net_optimizer,
        texture_discriminator_optimizer,

        device,
        num_epoch,
        args
        ):
        self.model_name = model_name
        self.args = args

        # model
        self.HMR = HMR
        self.context_encoder_net = context_encoder_net
        self.jigsaw_puzzle_net = jigsaw_puzzle_net
        self.rotation_net = rotation_net
        self.discriminator = discriminator
        self.texture_net = texture_net
        self.img_renderer = img_renderer
        self.seg_renderer = seg_renderer
        self.texture_discriminator = texture_discriminator

        # device
        self.device = device

        # dataloader
        self.train_dataloader = train_dataloader
        if test_dataloader:
            self.test_dataloader = test_dataloader
        if test_dataloader_h36m:
            self.test_dataloader_h36m = test_dataloader_h36m
        if test_dataloader_3dpw:
            self.test_dataloader_3dpw = test_dataloader_3dpw
        if test_dataloader_lsp:
            self.test_dataloader_lsp = test_dataloader_lsp
        # loss
        self.loss_fn_BCE = loss_fn_BCE
        self.loss_fn_CE = loss_fn_CE
        self.loss_fn_MSE = loss_fn_MSE
        self.loss_fn_keypoints = loss_fn_keypoints
        self.loss_fn_mask = loss_fn_mask

        # optimizer
        self.HMR_optimizer_all = HMR_optimizer_all
        self.HMR_scheduler_all = HMR_scheduler_all
        self.discriminator_optimizer = discriminator_optimizer
        self.context_encoder_optimizer = context_encoder_optimizer
        self.jigsaw_puzzle_optimizer = jigsaw_puzzle_optimizer
        self.rotation_optimizer = rotation_optimizer
        self.texture_net_optimizer = texture_net_optimizer
        self.texture_discriminator_optimizer = texture_discriminator_optimizer

        # Valiable
        self.num_epoch = num_epoch
        self.freq_print = args.freq_print
        self.num_patch = 4
        self.tex_size = args.tex_size
        today = datetime.datetime.now()
        self.today = datetime.datetime.strftime(today, "%y%m%d_%H%M%S")
        self.output_dir = self.args.output_dir if self.args.output_dir else self.today

        # smpl and J_regressor
        self.smpl_neutral = SMPL("data/vibe_data",
                        create_transl=False).to(self.device)
        self.smpl_male = SMPL("data/vibe_data",
                     gender='male',
                     create_transl=False).to(self.device)
        self.smpl_female = SMPL("data/vibe_data",
                       gender='female',
                       create_transl=False).to(self.device)
        self.J_regressor = np.load("data/vibe_data/J_regressor_h36m.npy")
        self.J_regressor_torch = torch.from_numpy(self.J_regressor).float()
        parts_texture = np.load("data/vertex_texture.npy")
        self.parts_texture = torch.from_numpy(parts_texture).to(self.device).float()
        self.cube_parts = torch.FloatTensor(np.load("data/cube_parts.npy")).to(self.device)

    def train(self):
        print("===================Train===================\nEpoch {} Start".format(self.epoch+1))
        train_template = \
            'Epoch: {}/{} | Batch_idx: {}/{} | ' \
            'loss_DC: {losses_DC.val:.4f} ({losses_DC.avg:.4f}) | loss_CE: {losses_CE.val:.4f} ({losses_CE.avg:.4f}) | ' \
            'loss_JP: {losses_JP.val:.4f} ({losses_JP.avg:.4f}) | acc_JP: {acces_JP.val:.4f} ({acces_JP.avg:.4f}) | ' \
            'loss_ROT: {losses_ROT.val:.4f} ({losses_ROT.avg:.4f}) | acc_ROT: {acces_ROT.val:.4f} ({acces_ROT.avg:.4f}) | ' \
            'loss_Texture: {losses_texture_ori_img.val:.4f} ({losses_texture_ori_img.avg:.4f}) | ' \
            'loss_Seg: {losses_seg.val:.4f} ({losses_seg.avg:.4f}) | ' \
            'loss_Texture_Total: {losses_texture_total.val:.4f} ({losses_texture_total.avg:.4f}) | ' \
            'loss_disc_e: {losses_disc_e.val:.4f} ({losses_disc_e.avg:.4f}) | ' \
            'loss_disc_d: {losses_disc.val:.4f} ({losses_disc.avg:.4f}) | ' \
            'loss_disc_real: {losses_disc_real.val:.4f} ({losses_disc_real.avg:.4f}) | ' \
            'loss_disc_fake: {losses_disc_fake.val:.4f} ({losses_disc_fake.avg:.4f}) | ' \
            'loss_HMR_3task: {losses_HMR_3task.val:.4f} ({losses_HMR_3task.avg:.4f}) | ' \
            'loss_HMR_joints3d: {losses_HMR_joints3d.val:.4f} ({losses_HMR_joints3d.avg:.4f}) | ' \
            'loss_joints: {losses_joints.val:.4f} ({losses_joints.avg:.4f}) | ' \
            'MPJPE: {train_MPJPE.val:.4f} ({train_MPJPE.avg:.4f}) | ' \
            'PA_MPJPE: {train_PA_MPJPE.val:.4f} ({train_PA_MPJPE.avg:.4f}) | ' \
            'loss_total: {losses_total.val:.4f} ({losses_total.avg:.4f}) | ' \
            'Batch duration: {duration}'

        batch_start = time.time()

        self.losses_DC = AverageMeter() # Discriminator Loss
        self.losses_CE = AverageMeter() # Context Encoder Loss
        self.losses_JP = AverageMeter() # Jigsaw Puzzle Loss
        self.acces_JP = AverageMeter() # Jigsaw Puzzle Accuracy
        self.losses_ROT = AverageMeter() # Rotation Loss
        self.acces_ROT = AverageMeter() # Rotation Accuracy
        self.losses_texture_ori_img = AverageMeter() # Texture Loss
        self.losses_seg = AverageMeter() # Segmentation Loss
        self.losses_texture_total = AverageMeter() # Texture Loss + Segmentation Loss
        self.losses_HMR_3task = AverageMeter() # Discriminator Loss + Context Encoder Loss + Roation Loss
        self.losses_disc_e = AverageMeter() # Encoder Discriminator Loss for rendering img
        self.losses_disc = AverageMeter() # Real + Fake Discriminator Loss
        self.losses_disc_real = AverageMeter() # Discriminator Loss for Real
        self.losses_disc_fake = AverageMeter() # Discriminator Loss for Fake
        self.losses_total = AverageMeter() # Total Sum Loss
        self.losses_joints = AverageMeter() # 2d + 3d joints loss
        self.losses_HMR_joints3d = AverageMeter() # 3D joints loss
        self.train_MPJPE = AverageMeter() # MPJPE
        self.train_PA_MPJPE = AverageMeter() # PA_MPJPE
        
        len_train_dataloader = len(self.train_dataloader)

        for batch_idx, item in tqdm.tqdm(enumerate(self.train_dataloader), desc='Train {}/{}'.format(self.epoch+1, self.num_epoch), total=len(self.train_dataloader)):
            img = item['img'].to(self.device)
            black_img = item['black_img'].to(self.device)
            context_encoder_input = item['context_encoder_input'].to(self.device)
            center_crop_img = item['center_crop_img'].to(self.device)
            jigsaw_input = item['jigsaw_input'].to(self.device)
            rotation_img = item['rotation_input'].to(self.device)
            jigsaw_order = item['jigsaw_order'].to(self.device)
            rotation_idx = item['rotation_idx'].to(self.device)
            joints3d = item['pose_3d'].to(self.device)
            has_joints3d = item['has_pose_3d'].to(self.device)
            joints2d = item['keypoints'].to(self.device)
            batch_size = img.shape[0]

            gt_mask = item['gt_mask'].to(self.device)
            has_mask = item['has_mask'].to(self.device)

            self.zeros = torch.zeros([batch_size, 1]).to(self.device)
            self.ones = torch.ones([batch_size, 1]).to(self.device)

            faces = get_smpl_faces().astype(np.int32)
            faces = torch.from_numpy(faces).to(self.device)
            faces = faces.expand((batch_size, -1, -1))

            joint_mapper_gt = constants.J24_TO_J17 if self.args.test_dataset == 'mpi-inf-3dhp' else constants.J24_TO_J14

            if self.epoch < self.args.first_stage_nEpoch: # Epoch 0~9
                # Training 3 Task net
                # Discriminator, Context Encoder, Jigsaw Puzzle, Rotation net
                self.HMR.eval()
                self.context_encoder_net.train()
                self.discriminator.train()
                self.jigsaw_puzzle_net.train()
                self.rotation_net.train()
                output_ce_224 = \
                    train_only_3task_network(
                        self.HMR,
                        self.context_encoder_net,
                        self.discriminator,
                        self.jigsaw_puzzle_net,
                        self.rotation_net,

                        self.loss_fn_BCE,
                        self.loss_fn_MSE,
                        self.loss_fn_CE,

                        self.losses_CE,
                        self.losses_DC,
                        self.acces_JP,
                        self.losses_JP,
                        self.acces_ROT,
                        self.losses_ROT,

                        self.discriminator_optimizer,
                        self.context_encoder_optimizer,
                        self.jigsaw_puzzle_optimizer,
                        self.rotation_optimizer,

                        img,
                        context_encoder_input, 
                        center_crop_img, 
                        jigsaw_input,
                        jigsaw_order,
                        rotation_img,
                        rotation_idx,
                        
                        self.num_patch,
                        self.ones,
                        self.zeros,
                        batch_size,
                    )

                # Training Texture Net
                self.texture_net.train()
                output_train_texture_net = \
                    train_texture_net(
                        self.HMR,
                        self.texture_net,
                        self.img_renderer,

                        self.loss_fn_MSE,
                        self.loss_fn_mask,

                        self.losses_texture_ori_img,
                        self.losses_seg,
                        self.losses_texture_total,

                        self.texture_net_optimizer,

                        img,
                        black_img,

                        batch_size,
                        self.args,
                        gt_mask,
                        has_mask,
                        train_first_stage=True
                    )

                mask = output_train_texture_net[0]
                detach_images = output_train_texture_net[1]
                rendering = output_train_texture_net[2]
                vertices = output_train_texture_net[3]

            # train hmr & texture net using 3task, rendering, segmentation and gan loss (or joints)
            else: # Epoch 10~19
                self.HMR.eval()
                self.context_encoder_net.eval()
                self.discriminator.eval()
                self.jigsaw_puzzle_net.eval()
                self.rotation_net.eval()
                self.texture_net.train()
                self.texture_discriminator.train()

                # Training Mesh network (HMR) using 3 Task Loss
                loss_HMR, output_ce_224 = \
                    train_hmr_using_3task(
                        self.HMR,
                        self.context_encoder_net,
                        self.discriminator,
                        self.jigsaw_puzzle_net,
                        self.rotation_net,

                        self.loss_fn_BCE,
                        self.loss_fn_MSE,
                        self.loss_fn_CE,

                        self.losses_CE,
                        self.acces_JP,
                        self.losses_JP,
                        self.acces_ROT,
                        self.losses_ROT,
                        self.losses_HMR_3task,

                        img,
                        context_encoder_input, 
                        center_crop_img, 
                        jigsaw_input,
                        jigsaw_order,
                        rotation_img,
                        rotation_idx,

                        self.num_patch,
                        self.ones,
                        self.zeros,
                        batch_size,
                        self.args
                    )
                
                loss_all = loss_HMR

                # Trainign Texture net
                output_train_texture_net = \
                    train_texture_net(
                        self.HMR,
                        self.texture_net,
                        self.img_renderer,

                        self.loss_fn_MSE,
                        self.loss_fn_mask,

                        self.losses_texture_ori_img,
                        self.losses_seg,
                        self.losses_texture_total,

                        self.texture_net_optimizer,

                        img,
                        black_img,

                        batch_size,
                        self.args,
                        gt_mask,
                        has_mask,
                        train_first_stage=False
                    )
                
                texture_loss = output_train_texture_net[0]
                loss_all += texture_loss

                mask = output_train_texture_net[1]
                detach_images = output_train_texture_net[2]
                rendering = output_train_texture_net[3]
                vertices = output_train_texture_net[4]
                
                # Trining HMR using adversarial loss
                e_disc_loss, d_disc_loss, rendering_bg = \
                    train_hmr_using_adv_loss(
                        self.HMR,
                        self.texture_discriminator,
                        self.texture_net,
                        self.img_renderer,

                        self.losses_disc_e,
                        self.losses_disc,
                        self.losses_disc_real,
                        self.losses_disc_fake,

                        img,

                        batch_size,
                    )
                
                loss_all += e_disc_loss
                self.texture_discriminator_optimizer.zero_grad()    
                d_disc_loss.backward()
                self.texture_discriminator_optimizer.step() 
                
                if not self.args.self_supervised:
                    # Training Mesh network (HMR) using joints
                    joints_loss, mpjpe, pa_mpjpe, num_data = train_hmr_using_joints(
                        self.HMR,

                        self.loss_fn_keypoints,

                        self.losses_HMR_joints3d,

                        img,
                        joints2d,
                        joints3d,
                        has_joints3d,

                        joint_mapper_gt,
                        batch_size,
                        self.device,
                        self.args,
                    )
                    
                    loss_all += joints_loss

                    self.losses_joints.update(joints_loss.item(), num_data)
                    self.train_MPJPE.update(mpjpe.item(), num_data)
                    self.train_PA_MPJPE.update(pa_mpjpe.item(), num_data)

                self.HMR_optimizer_all.zero_grad()
                self.losses_total.update(loss_all.item(), batch_size)
                loss_all.backward()
                self.HMR_optimizer_all.step()

            if (batch_idx==0 or (batch_idx+1)%self.args.freq_print==0):
                for i in range(10 if batch_size > 10 else batch_size):
                    img_dict = dict()
                    ### original img ###
                    img_dict["orig_img.jpg"] = img[i]
                    
                    ### context encoder input img ###
                    img_dict["ce_input_img.jpg"] = context_encoder_input[i].clone().detach()
                    ### center crop img for CE  ###
                    img_dict["center_crop_img.jpg"] = center_crop_img[i].clone().detach()
                    ### output img of CE ###
                    img_dict["reconst_img.jpg"] = output_ce_224[i].clone().detach()
                    ### jigsaw input img ###
                    img_dict["jigsaw_input_img.jpg"] = jigsaw_input[i].clone().detach()
                    ### ratation input img ###
                    img_dict["rotation_input_img.jpg"] = rotation_img[i].clone().detach()
                    
                    ### texture img ###
                    img_dict["rendering.jpg"] = rendering[i].clone().detach()
                    ### segmentation img ###
                    img_dict["mask.jpg"] = mask[i].clone().detach()
                    ### detach img ###
                    img_dict["detach.jpg"] = detach_images[i].clone().detach()
                    ### Segmentation gt ###
                    img_dict["seg_gt.jpg"] = gt_mask[i].clone().detach()
                    if self.epoch >= self.args.first_stage_nEpoch:
                        ### rendering background ###
                        img_dict["rendering_bg.jpg"] = rendering_bg[i].clone().detach()

                    save_all_img(img_dict, self.output_dir, 
                                self.epoch+1, i+batch_idx)
                        
                    ### save mesh ###
                    if self.epoch >= self.args.first_stage_nEpoch:
                        _faces = faces[i].clone().detach()
                        _vertices = vertices[i].clone().detach()
                        save_mesh(
                            _vertices, _faces, 
                            self.output_dir,
                            self.epoch+1,
                            i,
                        )
            ### print train info while running in batch loop ###
            if (batch_idx+1) % self.freq_print == 0 or (batch_idx+1) == len_train_dataloader:
                train_template_filled = train_template.format(
                    self.epoch+1, self.num_epoch,
                    batch_idx+1, len(self.train_dataloader),
                    losses_DC=self.losses_DC,
                    losses_CE=self.losses_CE,
                    losses_JP=self.losses_JP,
                    acces_JP=self.acces_JP,
                    losses_ROT=self.losses_ROT,
                    acces_ROT=self.acces_ROT,
                    losses_texture_ori_img=self.losses_texture_ori_img,
                    losses_seg=self.losses_seg,
                    losses_texture_total=self.losses_texture_total,
                    losses_disc_e=self.losses_disc_e,
                    losses_disc=self.losses_disc,
                    losses_disc_real=self.losses_disc_real,
                    losses_disc_fake=self.losses_disc_fake,
                    losses_HMR_3task=self.losses_HMR_3task,
                    losses_HMR_joints3d=self.losses_HMR_joints3d,
                    losses_joints=self.losses_joints,
                    losses_total=self.losses_total,
                    train_MPJPE=self.train_MPJPE,
                    train_PA_MPJPE=self.train_PA_MPJPE,
                    duration=get_HHMMSS_from_second(seconds=(time.time()-batch_start))
                )
                print(train_template_filled)
                self.train_templates.append(train_template_filled)
                if (batch_idx+1) == len_train_dataloader:
                    self.train_templates.append("======================================================================")
                batch_start = time.time()
        
        ### save train info when one epoch is completed ###
        save_templates_info(self.train_templates, self.output_dir, "train_templates.txt")

        print("Train Time: {train_time}, Total Time: {total_time}".format(
                train_time=get_HHMMSS_from_second(seconds=(time.time()-self.epoch_start)),
                total_time=get_HHMMSS_from_second(seconds=(time.time()-self.total_start))))


    ### Evaluate ###
    def evaluate(self, test_dataloader, test_dataset_name, is_save_pth=True):
        joint_mapper_h36m = constants.H36M_TO_J17 if self.args.test_dataset == 'mpi-inf-3dhp' else constants.H36M_TO_J14
        joint_mapper_gt = constants.J24_TO_J17 if self.args.test_dataset == 'mpi-inf-3dhp' else constants.J24_TO_J14
        test_start = time.time()
        self.HMR.eval()

        if self.args.train == 0:
            if test_dataset_name == "h36m" or test_dataset_name == "3dpw":
                test_template_batch = \
                    '===================Test===================\n' \
                    'Batch: {}/{} | ' \
                    'Mpjpe: {mpjpe_average_meter.avg:.2f} ' \
                    'Rec_error: {pa_mpjpe_average_meter.avg:.2f} ' \
                    '\n=========================================='
            
            elif test_dataset_name == "lsp":
                test_template_batch = \
                    '===================Test===================\n' \
                    'Batch: {}/{} | ' \
                    'Part acc: {part_Acc_average_meter:.2f} ' \
                    'Part F1: {part_F1_average_meter:.2f} ' \
                    'FG-BG Acc: {Acc_average_meter:.2f} ' \
                    'FG-BG F1: {F1_average_meter:.2f} ' \
                    '\n=========================================='

        if test_dataset_name == "h36m":
            test_template = \
                '===================Test===================\n' \
                'Test Data: {} | ' \
                'Epoch: {}/{} | ' \
                'Mpjpe: {mpjpe_average_meter.avg:.2f} | ' \
                'Rec_error: {pa_mpjpe_average_meter.avg:.2f} | ' \
                'loss: {losses.avg:.5f} | ' \
                'Test Time: {test_time} | Epoch Time: {epoch_time} | ' \
                'Total Time: {total_time}' \
                '\n=========================================='

        elif test_dataset_name == "3dpw":
            test_template = \
                '===================Test===================\n' \
                'Test Data: {} | ' \
                'Epoch: {}/{} | ' \
                'Mpjpe: {mpjpe_average_meter.avg:.2f} | ' \
                'Rec_error: {pa_mpjpe_average_meter.avg:.2f} | ' \
                'Test Time: {test_time} | Epoch Time: {epoch_time} | ' \
                'Total Time: {total_time}' \
                '\n=========================================='
        
        elif test_dataset_name == "lsp":
            test_template = \
                '===================Test===================\n' \
                'Test Data: {} | ' \
                'Epoch: {}/{} | ' \
                'Part acc: {part_Acc_average_meter:.2f} | ' \
                'Part F1: {part_F1_average_meter:.2f} | ' \
                'FG-BG Acc: {Acc_average_meter:.2f} | ' \
                'FG-BG F1: {F1_average_meter:.2f} | ' \
                'Test Time: {test_time} | Epoch Time: {epoch_time} | ' \
                'Total Time: {total_time}' \
                '\n=========================================='

        self.mpjpe_average_meter = AverageMeter()
        self.pa_mpjpe_average_meter = AverageMeter()
        self.losses = AverageMeter()
        self.part_Acc_average_meter = 0
        self.part_F1_average_meter = 0
        self.Acc_average_meter = 0
        self.F1_average_meter = 0

        if self.args.train != 0:
            if test_dataset_name == "h36m":
                current_mpjpe = self.current_mpjpe_h36m
            elif test_dataset_name == "3dpw":
                current_mpjpe = self.current_mpjpe_3dpw
            elif test_dataset_name == "lsp":
                current_acc = self.current_acc_lsp
        
        batch_num = len(test_dataloader)
        
        accuracy = 0.
        parts_accuracy = 0.

        pixel_count = 0
        parts_pixel_count = 0
        
        tp = np.zeros((2,1))
        fp = np.zeros((2,1))
        fn = np.zeros((2,1))

        parts_tp = np.zeros((7,1))
        parts_fp = np.zeros((7,1))
        parts_fn = np.zeros((7,1))

        
        with torch.no_grad():
            for batch_idx, item in tqdm.tqdm(enumerate(test_dataloader), desc='{} Eval'.format(test_dataset_name), total=len(test_dataloader)):
                # Validation for early stopping
                if test_dataset_name == "h36m":
                    img = item["img"]
                    img = img.to(self.device)
                    batch_size = img.shape[0]

                    output = self.HMR(img, J_regressor=self.J_regressor)
                    output = output[-1]

                    pred_j3ds = output['kp_3d']
                    pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
                    pred_j3ds -= pred_pelvis

                    target_j3ds = item["pose_3d"]
                    target_j3ds = target_j3ds[:, joint_mapper_gt, :-1]
                    target_j3ds = target_j3ds.float().to(self.device)
                    target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0
                    target_j3ds -= target_pelvis
                    
                    loss = self.loss_fn_MSE(pred_j3ds, target_j3ds)
                    self.losses.update(loss.item(), batch_size)

                if test_dataset_name == "h36m" or test_dataset_name == "3dpw":
                    img = item["img"]
                    img = img.to(self.device)
                    batch_size = img.shape[0]
                    
                    output = self.HMR(img, J_regressor=self.J_regressor)
                    output = output[-1]

                    pred_j3ds = output['kp_3d']
                    pred_pelvis = (pred_j3ds[:,[2],:] + pred_j3ds[:,[3],:]) / 2.0
                    pred_j3ds -= pred_pelvis
                    pred_vertices = output["verts"]

                    faces = get_smpl_faces().astype(np.int32)
                    faces = torch.from_numpy(faces).to(self.device)
                    faces = faces.expand((batch_size, -1, -1))

                    J_regressor_batch = self.J_regressor_torch[None, :].expand(pred_vertices.shape[0], -1, -1).to(self.device)
                    if test_dataset_name == 'h36m':
                        target_j3ds = item["pose_3d"]
                        target_j3ds = target_j3ds[:, joint_mapper_gt, :-1]
                    else:
                        gt_pose = item["pose"].to(self.device)
                        gt_betas = item["betas"].to(self.device)
                        gender = item["gender"].to(self.device)
                        gt_vertices = self.smpl_male(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                        gt_vertices_female = self.smpl_female(global_orient=gt_pose[:,:3], body_pose=gt_pose[:,3:], betas=gt_betas).vertices 
                        gt_vertices[gender==1, :, :] = gt_vertices_female[gender==1, :, :]
                        target_j3ds = torch.matmul(J_regressor_batch, gt_vertices)
                        gt_pelvis = target_j3ds[:, [0],:].clone()
                        target_j3ds = target_j3ds[:, joint_mapper_h36m, :]
                        target_j3ds = target_j3ds - gt_pelvis 

                    target_j3ds = target_j3ds.float().to(self.device)
                    target_pelvis = (target_j3ds[:,[2],:] + target_j3ds[:,[3],:]) / 2.0
                    target_j3ds -= target_pelvis
                    errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                    S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
                    errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
                    m2mm = 1000

                    mpjpe = np.mean(errors) * m2mm
                    pa_mpjpe = np.mean(errors_pa) * m2mm
                    self.mpjpe_average_meter.update(mpjpe, batch_size)
                    self.pa_mpjpe_average_meter.update(pa_mpjpe, batch_size)

                elif test_dataset_name == "lsp":
                    annot_path = config.DATASET_FOLDERS['upi-s1h']
                    img = item["img"]
                    img = img.to(self.device)
                    batch_size = img.shape[0]
                    orig_shape = item["orig_shape"].cpu().numpy()
                    scale = item["scale"].cpu().numpy()
                    center = item["center"].cpu().numpy()
                    
                    output = self.HMR(img, J_regressor=self.J_regressor)
                    output = output[-1]
                    pred_vertices = output["verts"]
                    cam = output['theta'][:, :3]

                    faces = get_smpl_faces().astype(np.int32)
                    faces = torch.from_numpy(faces).to(self.device)
                    faces = faces.expand((batch_size, -1, -1))

                    mask, parts = self.seg_renderer(pred_vertices, cam)
                    save_gt_parts = []
                    save_gt_seg = []

                    for i in range(batch_size):
                        # After rendering, convert imate back to original resolution
                        pred_mask = uncrop(mask[i].cpu().numpy(), center[i], scale[i], orig_shape[i]) > 0
                        # Load gt mask
                        gt_mask = cv2.imread(os.path.join(annot_path, item['maskname'][i]), 0) > 0
                        if batch_idx == 0:
                            save_gt_seg.append(gt_mask)
                        # Evaluation consistent with the original UP-3D code
                        accuracy += (gt_mask == pred_mask).sum()
                        pixel_count += np.prod(np.array(gt_mask.shape))
                        for c in range(2):
                            cgt = gt_mask == c
                            cpred = pred_mask == c
                            tp[c] += (cgt & cpred).sum()
                            fp[c] +=  (~cgt & cpred).sum()
                            fn[c] +=  (cgt & ~cpred).sum()
                        f1 = 2 * tp / (2 * tp + fp + fn)

                    for i in range(batch_size):
                        pred_parts = uncrop(parts[i].cpu().numpy().astype(np.uint8), center[i], scale[i], orig_shape[i])
                        # Load gt part segmentation
                        gt_parts = cv2.imread(os.path.join(annot_path, item['partname'][i]), 0)
                        if batch_idx == 0:
                            save_gt_parts.append(gt_parts)
                        # Evaluation consistent with the original UP-3D code
                        # 6 parts + background
                        for c in range(7):
                            cgt = gt_parts == c
                            cpred = pred_parts == c
                            cpred[gt_parts == 255] = 0
                            parts_tp[c] += (cgt & cpred).sum()
                            parts_fp[c] +=  (~cgt & cpred).sum()
                            parts_fn[c] +=  (cgt & ~cpred).sum()
                        gt_parts[gt_parts == 255] = 0
                        pred_parts[pred_parts == 255] = 0
                        parts_f1 = 2 * parts_tp / (2 * parts_tp + parts_fp + parts_fn)
                        parts_accuracy += (gt_parts == pred_parts).sum()
                        parts_pixel_count += np.prod(np.array(gt_parts.shape))

                    self.part_Acc_average_meter = (parts_accuracy / parts_pixel_count) * 100
                    self.part_F1_average_meter = parts_f1[[0,1,2,3,4,5,6]].mean()

                    self.Acc_average_meter = (accuracy / pixel_count) * 100
                    self.F1_average_meter = f1.mean()

                if batch_idx == 0:
                    for i in range(10 if batch_size > 10 else batch_size):
                        img_dict = dict()
                        img_dict["orig_img.jpg"] = img[i]

                        if test_dataset_name == "h36m" or test_dataset_name == "3dpw":
                            save_joints3d_img(
                                target_j3ds[i], 
                                pred_j3ds[i], 
                                self.output_dir, 
                                self.epoch+1,
                                test_dataset=test_dataset_name,
                                test_idx=i
                            )
                            save_mesh(
                                pred_vertices[i], 
                                faces[i], 
                                self.output_dir, 
                                self.epoch+1,
                                test_dataset=test_dataset_name,
                                test_idx=i
                            )

                        elif test_dataset_name == "lsp":
                            img_dict["mask.jpg"] = mask[i].cpu().numpy()
                            img_dict["parts.jpg"] = parts[i].cpu().numpy()
                            img_dict["gt_parts.jpg"] = save_gt_parts[i]
                            img_dict["gt_seg.jpg"] = save_gt_seg[i]

                        save_all_img(img_dict, self.output_dir, self.epoch+1, test_dataset=test_dataset_name, test_idx=i)
                
                if self.args.train == 0 and (batch_idx+1) % self.args.freq_print_test == 0:
                    if test_dataset_name == "h36m" or test_dataset_name == "3dpw":
                        test_template_batch_filled = test_template_batch.format(
                            batch_idx+1,
                            batch_num,
                            mpjpe_average_meter=self.mpjpe_average_meter,
                            pa_mpjpe_average_meter=self.pa_mpjpe_average_meter,
                        )
                    
                    elif test_dataset_name == "lsp":
                        test_template_batch_filled = test_template_batch.format(
                            batch_idx+1,
                            batch_num,
                            part_Acc_average_meter=self.part_Acc_average_meter,
                            part_F1_average_meter=self.part_F1_average_meter,
                            Acc_average_meter=self.Acc_average_meter,
                            F1_average_meter=self.F1_average_meter,
                        )
                    print(test_template_batch_filled)

            if test_dataset_name == "h36m" or test_dataset_name == "3dpw":
                test_template_filled = test_template.format(
                    test_dataset_name,
                    self.epoch+1,
                    self.num_epoch,
                    mpjpe_average_meter=self.mpjpe_average_meter,
                    pa_mpjpe_average_meter=self.pa_mpjpe_average_meter,
                    losses=self.losses,
                    test_time=get_HHMMSS_from_second(seconds=(time.time()-test_start)),
                    epoch_time=get_HHMMSS_from_second(seconds=(time.time()-self.epoch_start)),
                    total_time=get_HHMMSS_from_second(seconds=(time.time()-self.total_start))
                )

            elif test_dataset_name == "3dpw":
                test_template_filled = test_template.format(
                    test_dataset_name,
                    self.epoch+1,
                    self.num_epoch,
                    mpjpe_average_meter=self.mpjpe_average_meter,
                    pa_mpjpe_average_meter=self.pa_mpjpe_average_meter,
                    test_time=get_HHMMSS_from_second(seconds=(time.time()-test_start)),
                    epoch_time=get_HHMMSS_from_second(seconds=(time.time()-self.epoch_start)),
                    total_time=get_HHMMSS_from_second(seconds=(time.time()-self.total_start))
                )

            elif test_dataset_name == "lsp":
                test_template_filled = test_template.format(
                    test_dataset_name,
                    self.epoch+1,
                    self.num_epoch,
                    part_Acc_average_meter=self.part_Acc_average_meter,
                    part_F1_average_meter=self.part_F1_average_meter,
                    Acc_average_meter=self.Acc_average_meter,
                    F1_average_meter=self.F1_average_meter,
                    test_time=get_HHMMSS_from_second(seconds=(time.time()-test_start)),
                    epoch_time=get_HHMMSS_from_second(seconds=(time.time()-self.epoch_start)),
                    total_time=get_HHMMSS_from_second(seconds=(time.time()-self.total_start))
                )
            print(test_template_filled)
        
            # save test templates info txt file
            if self.args.train != 0:
                if test_dataset_name == "h36m":
                    self.test_templates_h36m.append(test_template_filled)
                    templates_filename = "test_templates_h36m.txt"
                    save_templates_info(self.test_templates_h36m, self.output_dir, templates_filename)
                
                elif test_dataset_name == "3dpw":
                    self.test_templates_3dpw.append(test_template_filled)
                    templates_filename = "test_templates_3dpw.txt"
                    save_templates_info(self.test_templates_3dpw, self.output_dir, templates_filename)

                elif test_dataset_name == "lsp":
                    self.test_templates_lsp.append(test_template_filled)
                    templates_filename = "test_templates_lsp.txt"
                    save_templates_info(self.test_templates_lsp, self.output_dir, templates_filename)
            
            else:
                self.test_templates.append(test_template_filled)
                templates_filename = "test_templates.txt"
                save_templates_info(self.test_templates, self.output_dir, templates_filename)

        # save pth file
        if is_save_pth:
            if self.epoch >= self.args.first_stage_nEpoch:
                if test_dataset_name == "h36m" or test_dataset_name == "3dpw":
                    # Save best pth
                    if self.mpjpe_average_meter.avg < current_mpjpe:
                        print("MPJPE changes from {:.4f} to {:.4f}".format(current_mpjpe, self.mpjpe_average_meter.avg))
                        self.save_checkpoint_all(test_dataset_name, best=True)
                    else:
                        print("MPJPE doesn't change {:.4f}".format(current_mpjpe))
                elif test_dataset_name == "lsp":
                    # Save best pth
                    if self.Acc_average_meter > current_acc:
                        print("ACC changes from {:.4f} to {:.4f}".format(current_acc, self.Acc_average_meter))
                        self.save_checkpoint_all(test_dataset_name, best=True)
                    else:
                        print("ACC doesn't change {:.4f}".format(current_acc))

            # Save lastest pth
            if self.args.save_pth_all_epoch:
                self.save_checkpoint_all(test_dataset_name, save_all_epoch=True)
            else:
                self.save_checkpoint_all(test_dataset_name)


    def save_checkpoint_all(self, test_dataset_name, best=False, save_all_epoch=False):
        """
            save pth file
        """
        filename = "best_{}_{}.pth" if best else "{}.pth"
        if save_all_epoch and not best:
            filename = "{}_{}_epoch"+str(self.epoch+1)+".pth"

        ### 3task network save pth, texture net save pth ###
        if self.epoch < self.args.first_stage_nEpoch:
            save_checkpoint({
                "state_dict": self.context_encoder_net.state_dict(),
                "loss": self.losses_CE.avg,
                "optimizer": self.context_encoder_optimizer.state_dict()
            }, 
            self.output_dir,
            filename.format("context_encoder_net", test_dataset_name))
            
            save_checkpoint({
                "state_dict": self.discriminator.state_dict(),
                "loss": self.losses_DC.avg,
                "optimizer": self.discriminator_optimizer.state_dict()
            }, 
            self.output_dir,
            filename.format("discriminator", test_dataset_name))

            save_checkpoint({
                "state_dict": self.jigsaw_puzzle_net.state_dict(),
                "accuracy": self.acces_JP.avg,
                "optimizer": self.jigsaw_puzzle_optimizer.state_dict()
            }, 
            self.output_dir,
            filename.format("jigsaw_puzzle_net", test_dataset_name))

            save_checkpoint({
                "state_dict": self.rotation_net.state_dict(),
                "accuracy": self.acces_ROT.avg,
                "optimizer": self.rotation_optimizer.state_dict()
            }, 
            self.output_dir,
            filename.format("rotation_net", test_dataset_name))

            save_checkpoint({
                "state_dict": self.texture_net.state_dict(),
                "loss": self.losses_texture_ori_img.avg,
                "optimizer": self.texture_net_optimizer.state_dict(),
            }, 
            self.output_dir,
            filename.format("texture_net", test_dataset_name))

        ### hmr save pth ###
        else:
            save_checkpoint({
                "state_dict": self.HMR.state_dict(),
                "loss_joints3d": self.losses_HMR_joints3d.avg,
                "loss_3task": self.losses_HMR_3task.avg,
                "optimizer_joints": self.HMR_optimizer_all.state_dict(),
            }, 
            self.output_dir,
            filename.format("hmr", test_dataset_name))

            save_checkpoint({
                "state_dict": self.texture_discriminator.state_dict(),
                "loss": self.losses_disc.avg,
                "optimizer": self.texture_discriminator_optimizer.state_dict()
            }, 
            self.output_dir,
            filename.format("texture_discriminator", test_dataset_name))
        
        if best:
            if test_dataset_name == "h36m":    
                self.current_mpjpe_h36m = self.mpjpe_average_meter.avg
            elif test_dataset_name == "3dpw":    
                self.current_mpjpe_3dpw = self.mpjpe_average_meter.avg
            elif test_dataset_name == "lsp":    
                self.current_acc_lsp = self.Acc_average_meter


    def fit(self):
        if self.args.train != 0:
            self.current_mpjpe_h36m = math.inf
            self.current_mpjpe_3dpw = math.inf
            self.current_acc_lsp = 0

            self.train_loss = math.inf

            self.total_start = time.time()
            self.train_templates = list()
            self.test_templates_h36m = list()
            self.test_templates_3dpw = list()
            self.test_templates_lsp = list()
            
            for epoch in range(self.num_epoch):
                self.epoch = epoch
                self.epoch_start = time.time()
                if epoch == 0 and self.args.first_eval:
                    self.epoch = -1
                    self.evaluate(self.test_dataloader_h36m, test_dataset_name="h36m", is_save_pth=False)
                    # self.evaluate(self.test_dataloader_3dpw, test_dataset_name="3dpw", is_save_pth=False)
                    # self.evaluate(self.test_dataloader_lsp, test_dataset_name="lsp", is_save_pth=False)
                    self.epoch = epoch
                print("HMR_optimizer_joints lr:", self.HMR_scheduler_all.get_lr())

                if self.epoch == self.args.first_stage_nEpoch:
                    texture_discriminator_checkpoint = torch.load(os.path.join("results", self.args.output_dir, "save_pth", "discriminator.pth"), map_location=self.device)
                    self.texture_discriminator.load_state_dict(texture_discriminator_checkpoint["state_dict"])

                self.train()

                if self.epoch >= self.args.first_stage_nEpoch:
                    self.HMR_scheduler_all.step()
                if (epoch+1)%self.args.freq_eval == 0 or (epoch+1) == self.num_epoch:
                    self.evaluate(self.test_dataloader_h36m, test_dataset_name="h36m")
                    # self.evaluate(self.test_dataloader_3dpw, test_dataset_name="3dpw")
                    # self.evaluate(self.test_dataloader_lsp, test_dataset_name="lsp")

        else:
            self.epoch = 0
            self.num_epoch = 1
            self.total_start = time.time()
            self.epoch_start = time.time()
            self.test_templates = list()
            
            self.evaluate(self.test_dataloader, test_dataset_name=self.args.test_dataset.replace("-p2", ""), is_save_pth=False)