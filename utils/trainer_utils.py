import time, datetime
import numpy as np
import os
import os.path as osp
import torch
import torchvision 
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import torch.nn.functional as F
import cv2
import glob
import random

from lib.utils.eval_utils import (
    batch_compute_similarity_transform_torch,
)
from lib.utils.geometry import batch_rodrigues


def inverse_normalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def normalize(tensor):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count


def get_HHMMSS_from_second(seconds):
    HHMMSS = time.strftime("%H:%M:%S", time.gmtime(seconds))
    return HHMMSS


def save_checkpoint(state, output_dir, filename):
    """ save model parameter """

    save_path = "results/{}/save_pth".format(output_dir)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    
    torch.save(state, osp.join(save_path, filename))


def save_img(img, output_dir, filename, epoch=None, img_path=None, test_dataset=None, test_idx=None, vflip=False, hflip=False):
    """ save image """
    if isinstance(img, torch.Tensor):
        img = img.cpu()

    if isinstance(img_path, int):
        img_path = str(img_path)
        root = "results/{}/save_output_train".format(output_dir)
        epoch = "epoch"+str(epoch)
        save_path = osp.join(root, epoch, img_path)

    elif epoch and img_path:
        dataset = img_path.split("/")[2]
        root = "results/{}/save_output_train/{}".format(output_dir, dataset)
        epoch = "epoch"+str(epoch)
        save_path = osp.join(root, epoch, ("_".join(img_path.split("/")[-2:])).replace(".jpg", ""))

    elif test_dataset:
        save_path = "results/{}/eval_data/{}_{}/{}".format(output_dir, test_dataset, epoch, test_idx)

    if not osp.exists(save_path):
        os.makedirs(save_path)

    # For jigsaw image with tile
    if len(img.shape) == 4: # tile, channel, width, height
        for tile_idx, _img_tile in enumerate(img):
            _filename = "_".join([filename.replace(".jpg", ""), str(tile_idx)]) + ".jpg"
            
            # _img = np.transpose(vutils.make_grid(_img_tile, normalize=True), (1, 2, 0))
            # _img = _img.numpy()
            _img = inverse_normalize(_img_tile).detach().numpy()
            _img = np.transpose(_img, (1, 2, 0))
            if vflip:
                _img = _img[::-1, :, :]
            if hflip:
                _img = _img[:, ::-1, :] 
            plt.imsave(osp.join(save_path, _filename), _img)

    # For a image
    elif len(img.shape) == 3: # channel, width, height
        if isinstance(img, torch.Tensor):
            # if filename == "reconst_img.jpg":
            #     img = np.transpose(vutils.make_grid(img, normalize=True), (1, 2, 0))
            #     img = img.numpy()
            #     plt.imsave(osp.join(save_path, filename), img)
            if filename in ["detach.jpg", "rendering.jpg", "rendering_ren.jpg", "rendering_bg.jpg", "rendering1.jpg", "rendering_ren1.jpg",
                            "rendering2.jpg", "rendering_ren2.jpg", "rendering3.jpg", "rendering_ren3.jpg",
                            "rendering4.jpg", "rendering_ren4.jpg"]:
                img = np.transpose(img, (1, 2, 0))
                img = img.numpy()
                cv2.imwrite(osp.join(save_path, filename), 255*img[:, :, ::-1])
            else:
                img = inverse_normalize(img).detach().numpy()
                img = np.transpose(img, (1, 2, 0))
                plt.imsave(osp.join(save_path, filename), img)
        # if vflip:
        #     img = img[::-1, :, :]
        # if hflip:
        #     img = img[:, ::-1, :]           

    elif len(img.shape) == 2: # width, height
        if isinstance(img, torch.Tensor):
            img = img[None, :, :]
            img = np.transpose(vutils.make_grid(img, normalize=True), (1, 2, 0))
            img = img.numpy()
            # img = inverse_normalize(img).detach().numpy()[:, :, None]
            # img = np.transpose(img, (1, 2, 0))
        if vflip:
            img = img[::-1, :, :]
        if hflip:
            img = img[:, ::-1, :]           
        
        plt.imsave(osp.join(save_path, filename), img)


def save_all_img(img_dict, output_dir, epoch=None, img_path=None, test_dataset=None, test_idx=None, vflip=False, hflip=False):
    """
        img_dict keys: filename, value: image 
    """
    for filename, img in img_dict.items():
        save_img(img, output_dir, filename, epoch, img_path, test_dataset, test_idx, vflip, hflip)


def save_mesh(verts, faces, output_dir, epoch, img_path=None, test_dataset=None, test_idx=None):
    """ save verts """
    filename = "mesh.obj"
    img = verts.cpu().numpy()
    faces = faces.cpu().numpy()
    
    if isinstance(img_path, int):
        img_path = str(img_path)
        root = "results/{}/save_output_train".format(output_dir)
        epoch = "epoch"+str(epoch)
        save_path = osp.join(root, epoch, img_path)

    elif test_dataset is not None and test_idx is not None:
        save_path = "results/{}/eval_data/{}_{}/{}".format(output_dir, test_dataset, epoch, test_idx)

    else:
        dataset = img_path.split("/")[2]
        root = "results/{}/save_output_train/{}".format(output_dir, dataset)
        epoch = "epoch"+str(epoch)
        save_path = osp.join(root, epoch, ("_".join(img_path.split("/")[-2:])).replace(".jpg", ""))
    
    if not osp.exists(save_path):
        os.makedirs(save_path)
    
    if len(verts.shape) == 2:
        with open(osp.join(save_path, filename), "w") as f:
            for verts_xyz in verts:
                f.write("v {} {} {}\n".format(verts_xyz[0], verts_xyz[1], verts_xyz[2]))
            
            for face in faces:
                f.write("f {} {} {}\n".format(face[0]+1, face[1]+1, face[2]+1))
    

def save_templates_info(test_templates, output_dir, filename):
    """ save templates
        save results information or train information """

    save_path = "results/{}/save_txt".format(output_dir)
    if not osp.exists(save_path):
        os.makedirs(save_path)
    
    _filename = osp.join(save_path, filename)

    with open(_filename, "w") as f:
        f.writelines("\n".join(test_templates))


def save_joints2d_img(gt_keypoints_2d, pred_keypoints_2d, output_dir, epoch, img_path):
    """
        save image of joints2d on 2d coordinate while training
    """
    line_list = [
        [ 0, 1 ],
        [ 1, 2 ],
        [ 3, 4 ],
        [ 4, 5 ],
        [ 6, 7 ],
        [ 7, 8 ],
        [ 8, 2 ],
        [ 8, 9 ],
        [ 9, 3 ],
        [ 2, 3 ],
        [ 8, 12],
        [ 9, 10],
        [12, 9 ],
        [10, 11],
        [12, 13],
    ]

    if isinstance(img_path, int):
        img_path = str(img_path)
        root = "results/{}/save_output_train".format(output_dir)
        epoch = "epoch"+str(epoch)
        save_path = osp.join(root, epoch, img_path)
    else:
        dataset = img_path.split("/")[2]
        root = "results/{}/save_output_train/{}".format(output_dir, dataset)
        epoch = "epoch"+str(epoch)
        save_path = osp.join(root, epoch, ("_".join(img_path.split("/")[-2:])).replace(".jpg", ""))
    
    if not osp.exists(save_path):
        os.makedirs(save_path)    

    plt.figure()
    ax = plt.subplot()
    ax.invert_yaxis()
    
    gt_2ds = gt_keypoints_2d.clone()
    gt_2ds = gt_2ds.cpu()
    
    pred_2ds = pred_keypoints_2d.clone()
    pred_2ds = pred_2ds.cpu().detach()

    for joint_idx, (gt_2d, pred_2d) in enumerate(zip(gt_2ds, pred_2ds)):
        ax.scatter(gt_2d[0], gt_2d[1], marker='o', s=2, c="r")
        ax.text(gt_2d[0], gt_2d[1], joint_idx+1)
        ax.scatter(pred_2d[0], pred_2d[1], marker='o', s=2, c="b")
        ax.text(pred_2d[0], pred_2d[1], joint_idx+1)

    for start_point, end_point in line_list:
        start_point = start_point
        end_point = end_point
        ax.plot([gt_2ds[start_point][0], gt_2ds[end_point][0]], [gt_2ds[start_point][1], gt_2ds[end_point][1]], "r", linewidth=2)
        ax.plot([pred_2ds[start_point][0], pred_2ds[end_point][0]], [pred_2ds[start_point][1], pred_2ds[end_point][1]], "b", linewidth=2)

    plt.savefig(osp.join(save_path, "joints2d.jpg"))
    plt.close()


def save_joints3d_img(gt_keypoints_3d, pred_keypoints_3d, output_dir, epoch=None, img_path=None, test_dataset=None, test_idx=None):
    """
        save image of joints3d on 3d coordinate while training
    """
    line_list = [
        [ 0, 1 ],
        [ 1, 2 ],
        [ 3, 4 ],
        [ 4, 5 ],
        [ 6, 7 ],
        [ 7, 8 ],
        [ 8, 2 ],
        [ 8, 9 ],
        [ 9, 3 ],
        [ 2, 3 ],
        [ 8, 12],
        [ 9, 10],
        [12, 9 ],
        [10, 11],
        [12, 13],
    ]
    if isinstance(img_path, int):
        img_path = str(img_path)
        root = "results/{}/save_output_train".format(output_dir)
        epoch = "epoch"+str(epoch)
        save_path = osp.join(root, epoch, img_path)
    
    elif epoch and img_path:
        dataset = img_path.split("/")[2]
        root = "results/{}/save_output_train/{}".format(output_dir, dataset)
        epoch = "epoch"+str(epoch)
        save_path = osp.join(root, epoch, ("_".join(img_path.split("/")[-2:])).replace(".jpg", ""))

    elif test_dataset is not None and test_idx is not None:
        save_path = "results/{}/eval_data/{}_{}/{}".format(output_dir, test_dataset, epoch, test_idx)

    if not osp.exists(save_path):
        os.makedirs(save_path)

    plt.figure()
    ax = plt.subplot(1, 1, 1, projection='3d')
    gt_3ds = gt_keypoints_3d.clone()
    gt_3ds = gt_3ds.cpu()
    pred_3ds = pred_keypoints_3d.clone()
    pred_3ds = pred_3ds.cpu().detach()
    for joint_idx, (gt_3d, pred_3d) in enumerate(zip(gt_3ds, pred_3ds)):
        ax.scatter(gt_3d[0], gt_3d[1], gt_3d[2], marker='o', s=2, c="r")
        ax.text(gt_3d[0], gt_3d[1], gt_3d[2], joint_idx+1)
        ax.scatter(pred_3d[0], pred_3d[1], pred_3d[2], marker='o', s=2, c="b")
        ax.text(pred_3d[0], pred_3d[1], pred_3d[2], joint_idx+1)

    for start_point, end_point in line_list:
        start_point = start_point
        end_point = end_point
        ax.plot(
            [gt_3ds[start_point][0], gt_3ds[end_point][0]], 
            [gt_3ds[start_point][1], gt_3ds[end_point][1]], 
            [gt_3ds[start_point][2], gt_3ds[end_point][2]], 
            color="r", 
            linewidth=2)
        ax.plot(
            [pred_3ds[start_point][0], pred_3ds[end_point][0]], 
            [pred_3ds[start_point][1], pred_3ds[end_point][1]], 
            [pred_3ds[start_point][2], pred_3ds[end_point][2]], 
            color="b", 
            linewidth=2)

    plt.savefig(osp.join(save_path, "joints3d.jpg"))
    plt.close()


def get_acc(output, label):
    batch_size = output.shape[0]
    pred = torch.argmax(output, dim=1)
    correct = (pred==label).sum().item()
    acc = correct/batch_size * 100
    return acc


def spin2h36m_joint(spins, device):
    """
        Get h36m 14 joints from spin 49 joints
    """
    convert_matrix = torch.zeros((14, 49)).to(device)
    h36m_index_list = [25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38]
    for idx, h36m_index in enumerate(h36m_index_list):
        convert_matrix[idx, h36m_index] = 1
    h36m_joints = torch.matmul(convert_matrix, spins)
    return h36m_joints


def get_spherical_coords(X):
    # X is N x 3
    rad = np.linalg.norm(X, axis=1)
    # Inclination
    theta = np.arccos(X[:, 2] / rad)
    # Azimuth
    phi = np.arctan2(X[:, 1], X[:, 0])

    # Normalize both to be between [-1, 1]
    vv = (theta / np.pi) * 2 - 1
    uu = ((phi + np.pi) / (2*np.pi)) * 2 - 1
    # Return N x 2
    return np.stack([uu, vv],1)
    

def compute_uvsampler(verts, faces, tex_size=2):
    """
    For this mesh, pre-computes the UV coordinates for
    F x T x T points.
    Returns F x T x T x 2
    """
    alpha = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    beta = np.arange(tex_size, dtype=np.float) / (tex_size-1)
    import itertools
    # Barycentric coordinate values
    coords = np.stack([p for p in itertools.product(*[alpha, beta])])
    vs = verts[faces]
    # Compute alpha, beta (this is the same order as NMR)
    v2 = vs[:, 2]
    v0v2 = vs[:, 0] - vs[:, 2]
    v1v2 = vs[:, 1] - vs[:, 2]    
    # F x 3 x T*2
    samples = np.dstack([v0v2, v1v2]).dot(coords.T) + v2.reshape(-1, 3, 1)    
    # F x T*2 x 3 points on the sphere 
    samples = np.transpose(samples, (0, 2, 1))

    # Now convert these to uv.
    uv = get_spherical_coords(samples.reshape(-1, 3))
    # uv = uv.reshape(-1, len(coords), 2)

    uv = uv.reshape(-1, tex_size, tex_size, 2)
    return uv

def train_only_3task_network(
        HMR,
        context_encoder_net,
        discriminator,
        jigsaw_puzzle_net,
        rotation_net,
        
        loss_fn_BCE,
        loss_fn_MSE,
        loss_fn_CE,

        losses_CE,
        losses_DC,
        acces_JP,
        losses_JP,
        acces_ROT,
        losses_ROT,

        discriminator_optimizer,
        context_encoder_optimizer,
        jigsaw_puzzle_optimizer,
        rotation_optimizer,

        img,
        context_encoder_input,
        center_crop_img, 
        jigsaw_input,
        jigsaw_order,
        rotation_img,
        rotation_idx,
        
        num_patch,
        ones,
        zeros,
        batch_size,
    ):
    ### Context Encoder ###
    # Update Discriminator
    
    feature_ce = HMR(context_encoder_input, return_only_features=True)
    feature_ce = feature_ce.reshape(-1, 2048, 1, 1)
    output_ce = context_encoder_net(feature_ce)
    output_ce_224 = context_encoder_input.clone()
    output_ce_224[:, :, 80:144, 80:144] = output_ce.clone()
    output_fake = discriminator(output_ce_224)
    output_real = discriminator(img)
    loss_BCE_fake = loss_fn_BCE(output_fake, zeros)
    loss_BCE_real = loss_fn_BCE(output_real, ones)
    loss_BCE = loss_BCE_fake + loss_BCE_real
    losses_DC.update(loss_BCE.item(), batch_size)
    discriminator_optimizer.zero_grad()
    loss_BCE.backward()
    discriminator_optimizer.step()

    # Update Decoder
    feature_ce = HMR(context_encoder_input, return_only_features=True)
    feature_ce = feature_ce.reshape(-1, 2048, 1, 1)
    output_ce = context_encoder_net(feature_ce)
    output_ce_224 = context_encoder_input.clone()
    output_ce_224[:, :, 80:144, 80:144] = output_ce.clone()
    output_fake = discriminator(output_ce_224)
    loss_BCE = loss_fn_BCE(output_fake, ones)
    loss_MSE = loss_fn_MSE(output_ce, center_crop_img)
    loss_ce = 0.001 * loss_BCE + 0.999 * loss_MSE
    losses_CE.update(loss_ce.item(), batch_size)
    context_encoder_optimizer.zero_grad()
    loss_ce.backward()
    context_encoder_optimizer.step()

    ### Jigsaw Puzzle ###
    # Update classifier
    _jigsaw_input = jigsaw_input.permute(1, 0, 2, 3, 4) # tile, batch, c, w, h
    feature_jp = list()
    for i in range(num_patch):
        feature_jp.append(HMR(_jigsaw_input[i], return_only_features=True))
    
    feature_jp.append(HMR(img, return_only_features=True))
    feature_jp = torch.cat(feature_jp, 1)
    output_jp = jigsaw_puzzle_net(feature_jp)
    acc_jp = get_acc(output_jp, jigsaw_order)
    acces_JP.update(acc_jp, batch_size)
    loss_jp = loss_fn_CE(output_jp, jigsaw_order)
    losses_JP.update(loss_jp.item(), batch_size)
    jigsaw_puzzle_optimizer.zero_grad()
    loss_jp.backward()
    jigsaw_puzzle_optimizer.step()
    
    ### Rotation ###
    # Update rotation net
    feature_rot = HMR(rotation_img, return_only_features=True)
    output_rot = rotation_net(feature_rot)
    acc_rot = get_acc(output_rot, rotation_idx)
    acces_ROT.update(acc_rot, batch_size)
    loss_rot = loss_fn_CE(output_rot, rotation_idx)
    losses_ROT.update(loss_rot, batch_size)
    rotation_optimizer.zero_grad()
    loss_rot.backward()
    rotation_optimizer.step()

    return output_ce_224

def train_hmr_using_3task(
        HMR,
        context_encoder_net,
        discriminator,
        jigsaw_puzzle_net,
        rotation_net,
        
        loss_fn_BCE,
        loss_fn_MSE,
        loss_fn_CE,

        losses_CE,
        acces_JP,
        losses_JP,
        acces_ROT,
        losses_ROT,
        losses_HMR_3task,

        img,
        context_encoder_input,
        center_crop_img, 
        jigsaw_input,
        jigsaw_order,
        rotation_img,
        rotation_idx,
        
        num_patch,
        ones,
        zeros,
        batch_size,
        args,
    ):
    # loss for HMR - ce
    feature_ce = HMR(context_encoder_input, return_only_features=True)
    feature_ce = feature_ce.reshape(-1, 2048, 1, 1)
    output_ce = context_encoder_net(feature_ce)
    output_ce_224 = context_encoder_input.clone()
    output_ce_224[:, :, 80:144, 80:144] = output_ce.clone()
    output_fake = discriminator(output_ce_224)
    loss_BCE = loss_fn_BCE(output_fake, ones)
    loss_MSE = loss_fn_MSE(output_ce, center_crop_img)
    loss_ce = 0.001 * loss_BCE + 0.999 * loss_MSE
    losses_CE.update(loss_ce.item(), batch_size)

    # loss for HMR - jp
    _jigsaw_input = jigsaw_input.permute(1, 0, 2, 3, 4) # tile, batch, c, w, h
    feature_jp = list()
    for i in range(num_patch):
        feature_jp.append(HMR(_jigsaw_input[i], return_only_features=True))
    feature_jp.append(HMR(img, return_only_features=True))
    feature_jp = torch.cat(feature_jp, 1)
    output_jp = jigsaw_puzzle_net(feature_jp)
    acc_jp = get_acc(output_jp, jigsaw_order)
    acces_JP.update(acc_jp, batch_size)
    loss_jp = loss_fn_CE(output_jp, jigsaw_order)
    losses_JP.update(loss_jp.item(), batch_size)

    # loss for HMR - rot
    feature_rot = HMR(rotation_img, return_only_features=True)
    output_rot = rotation_net(feature_rot)
    acc_rot = get_acc(output_rot, rotation_idx)
    acces_ROT.update(acc_rot, batch_size)
    loss_rot = loss_fn_CE(output_rot, rotation_idx)
    losses_ROT.update(loss_rot, batch_size)

    loss_HMR = args.ce_weight * loss_ce + args.jp_weight * loss_jp + args.rot_weight * loss_rot
    loss_HMR = args.total_weight * loss_HMR
    losses_HMR_3task.update(loss_HMR.item(), batch_size)
    return loss_HMR, output_ce_224

def train_hmr_using_joints(
        HMR,

        loss_fn_keypoints,

        losses_HMR_joints3d,

        img, 
        gt_keypoints_2d, 
        gt_keypoints_3d,
        has_joints3d,

        joint_mapper_gt,
        batch_size,
        device,
        args,
    ):
    
    ### training HMR resnet update using joints info
    output = HMR(img)
    output = output[-1]

    ### calcuate loss of 2d joints ###
    pred_keypoints_2d = output["kp_2d"]
    
    conf = gt_keypoints_2d[:, :, -1].unsqueeze(-1).clone()
    joints2d_loss = (conf * loss_fn_keypoints(pred_keypoints_2d, gt_keypoints_2d[:, :, :-1])).mean()
    joints2d_loss = 5*joints2d_loss

    ### calcuate loss of 3d joints ###
    pred_keypoints_3d = output["kp_3d"][:, 25:, :]
    
    conf = gt_keypoints_3d[:, :, -1].unsqueeze(-1).clone()
    gt_keypoints_3d = gt_keypoints_3d[:, :, :-1].clone()
    gt_keypoints_3d = gt_keypoints_3d[has_joints3d==1]
    conf = conf[has_joints3d==1]
    pred_keypoints_3d = pred_keypoints_3d[has_joints3d==1]
    
    if len(gt_keypoints_3d) > 0:
        gt_pelvis = (gt_keypoints_3d[:, 2, :] + gt_keypoints_3d[:, 3, :]) / 2
        gt_keypoints_3d = gt_keypoints_3d - gt_pelvis[:, None, :]
    
        pred_pelvis = (pred_keypoints_3d[:, 2, :] + pred_keypoints_3d[:, 3, :]) / 2
        pred_keypoints_3d = pred_keypoints_3d - pred_pelvis[:, None, :]
        joints3d_loss = (conf*loss_fn_keypoints(pred_keypoints_3d, gt_keypoints_3d)).mean()

        pred_j3ds = pred_keypoints_3d[:, joint_mapper_gt, :].clone().detach()
        target_j3ds = gt_keypoints_3d[:, joint_mapper_gt, :].clone().detach()

        errors = torch.sqrt(((pred_j3ds - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        S1_hat = batch_compute_similarity_transform_torch(pred_j3ds, target_j3ds)
        errors_pa = torch.sqrt(((S1_hat - target_j3ds) ** 2).sum(dim=-1)).mean(dim=-1).cpu().numpy()
        m2mm = 1000
        
        mpjpe = np.mean(errors) * m2mm
        pa_mpjpe = np.mean(errors_pa) * m2mm
        num_data = len(gt_keypoints_3d)
    else:
        joints3d_loss = torch.FloatTensor(1).fill_(0.).to(device).mean()
        mpjpe = np.array(0)
        pa_mpjpe = np.array(0)
        num_data = len(gt_keypoints_3d)
    
    total_loss = joints2d_loss

    joints3d_loss = 5*joints3d_loss
    total_loss += joints3d_loss
    losses_HMR_joints3d.update(joints3d_loss.item(), batch_size)

    total_loss *= 60
    return total_loss, mpjpe, pa_mpjpe, num_data


def train_texture_net(
        HMR,
        texture_net,
        img_renderer,

        loss_fn_MSE,
        loss_fn_mask,

        losses_texture_ori_img,
        losses_seg,
        losses_texture_total,

        texture_net_optimizer,

        img,
        black_img,

        batch_size,
        args,
        gt_mask,
        has_mask,
        train_first_stage
    ):  
    output = HMR(img)
    output = output[-1]
    vertices = output['verts']
    cam = output['theta'][:, :3]
    
    textures = texture_net(img)
    textures = textures.expand(-1, -1, 2, 2, 2, -1)
    
    mask = gt_mask.clone().detach()
    mask_est, rendering = img_renderer(vertices, cam, textures)
    
    valid_mask = mask > 0
    valid_mask = valid_mask[:, None, :, :].type(torch.int) 
    detach_images = img * valid_mask + black_img * (1-valid_mask)
    
    for i in range(batch_size):
        detach_images[i] = inverse_normalize(detach_images[i])

    #======================================================================================#
    # loss_texture => MSE loss (texture images with bg, original img)
    # loss_seg => MSE loss (segmentation images, target seg)
    # loss_texture_BCE => BCE loss (texture images with bg, Real(1))
    # loss_texture_total => SUM(loss_texture, loss_seg, loss_texture_BCE)
    #======================================================================================#
    loss_texture_ori_img = loss_fn_MSE(detach_images[has_mask==1], rendering[has_mask==1])
    losses_texture_ori_img.update(loss_texture_ori_img.item(), batch_size)
    loss_all = args.rendering_weight*loss_texture_ori_img
    
    if train_first_stage:
        texture_net_optimizer.zero_grad()
        loss_all.backward()
        texture_net_optimizer.step()
        return mask_est, detach_images, rendering, vertices
    else:
        _mask = mask_est[has_mask == 1]
        _gt_mask = gt_mask[has_mask == 1]
        
        if len(_mask) != 0 and len(_gt_mask) != 0:
            loss_seg = loss_fn_mask(_mask, _gt_mask)
            losses_seg.update(loss_seg.item(), batch_size)
            loss_all += args.seg_weight * loss_seg
        
        loss_all = args.texture_total_weight * loss_all
        losses_texture_total.update(loss_all.item(), batch_size)

        return loss_all, mask_est, detach_images, rendering, vertices


def train_hmr_using_adv_loss(
        HMR,
        texture_discriminator,
        texture_net,
        img_renderer,

        losses_disc_e,
        losses_disc,
        losses_disc_real,
        losses_disc_fake,

        img,

        batch_size,
    ):

    output = HMR(img)[-1]
    vertices = output['verts']
    cam = output['theta'][:, :3]

    textures = texture_net(img)
    textures = textures.expand(-1, -1, 2, 2, 2, -1)

    mask, rendering = img_renderer(vertices, cam, textures)
    
    rendering_img = rendering.clone()

    bg_idx = random.randint(1, 18)
    bg_list = glob.glob("/data/indoor_bg/train/LR/{}/color/*.png".format(bg_idx))

    cropped_bg_list = torch.zeros(batch_size, 3, 224, 224)
    for i in range(batch_size):
        random_idx = random.randint(0, len(bg_list)-1)
        bg_path = bg_list[random_idx]
        bg = cv2.imread(bg_path)
        bg_w, bg_h, _ = bg.shape
        h = w = 224
        rand_idx_w = int(np.random.randint(bg_w-w))
        rand_idx_h = int(np.random.randint(bg_h-h))
        cropped_bg = bg[rand_idx_w:rand_idx_w+w, rand_idx_h:rand_idx_h+h, :]/255.0
        cropped_bg = torch.from_numpy(cropped_bg).permute(2, 0, 1)
        cropped_bg_list[i] = cropped_bg

    cropped_bg_list = cropped_bg_list.to(rendering_img.device)
    valid_mask = mask > 0
    valid_mask = valid_mask[:, None, :, :].type(torch.int)

    rendering_img_input = valid_mask * rendering_img + (1-valid_mask) * cropped_bg_list
    rendering_bg = rendering_img_input.clone().detach()
    
    for i in range(batch_size):
        rendering_img_input[i] = normalize(rendering_img_input[i])
    
    e_disc_loss = batch_encoder_disc_l2_loss(texture_discriminator(rendering_img_input))
    losses_disc_e.update(e_disc_loss.item(), batch_size)

    fake_rendering_img_input = rendering_img_input.clone().detach()

    real = texture_discriminator(img)
    fake = texture_discriminator(fake_rendering_img_input)
    
    d_disc_real, d_disc_fake, d_disc_loss = batch_adv_disc_l2_loss(real, fake)
    losses_disc.update(d_disc_loss.item(), batch_size)
    losses_disc_real.update(d_disc_real.item(), batch_size)
    losses_disc_fake.update(d_disc_fake.item(), batch_size)
    return e_disc_loss, d_disc_loss, rendering_bg

def batch_encoder_disc_l2_loss(disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    k = disc_value.shape[0]
    return torch.sum((disc_value - 1.0) ** 2) * 1.0 / k


def batch_adv_disc_l2_loss(real_disc_value, fake_disc_value):
    '''
        Inputs:
            disc_value: N x 25
    '''
    ka = real_disc_value.shape[0]
    kb = fake_disc_value.shape[0]
    lb, la = torch.sum(fake_disc_value ** 2) / kb, torch.sum((real_disc_value - 1) ** 2) / ka
    return la, lb, la + lb