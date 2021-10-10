from __future__ import division

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import Normalize
import numpy as np
import cv2
from PIL import Image
import copy
import itertools
import os, os.path as osp
from os.path import join

import config
import constants
from utils.imutils import crop, flip_img, flip_pose, flip_kp, transform, rot_aa
import matplotlib.pyplot as plt

class BaseDataset(Dataset):
    """
    Base Dataset Class - Handles data loading and augmentation.
    Able to handle heterogeneous datasets (different annotations available for different datasets).
    You need to update the path to each dataset in utils/config.py.
    """

    def __init__(self, options, dataset, ignore_3d=False, use_augmentation=True, is_train=True):
        super(BaseDataset, self).__init__()
        self.dataset = dataset
        self.is_train = is_train
        self.options = options
        self.img_dir = config.DATASET_FOLDERS[dataset]

        self.normalize_img = Normalize(mean=constants.IMG_NORM_MEAN, std=constants.IMG_NORM_STD)
        self.data = np.load(config.DATASET_FILES[is_train][dataset])
        self.imgname = self.data['imgname']

        if not is_train:
            print("{} Test Num: {}".format(dataset, format(len(self.imgname), ",")))

        # Get paths to gt masks, if available
        try:
            self.maskname = self.data['maskname']
        except KeyError:
            pass
        try:
            self.partname = self.data['partname']
        except KeyError:
            pass

        # Bounding boxes are assumed to be in the center and scale format
        self.scale = self.data['scale']
        self.center = self.data['center']
        
        # If False, do not do augmentation
        self.use_augmentation = use_augmentation
        
        # Get gt SMPL parameters, if available
        try:
            self.pose = self.data['pose'].astype(np.float)
            self.betas = self.data['shape'].astype(np.float)
            if 'has_smpl' in self.data:
                self.has_smpl = self.data['has_smpl']
            else:
                self.has_smpl = np.ones(len(self.imgname))
        except KeyError:
            self.has_smpl = np.zeros(len(self.imgname))
        if ignore_3d:
            self.has_smpl = np.zeros(len(self.imgname))
        
        # Get gt 3D pose, if available
        try:
            self.pose_3d = self.data['S']
            self.has_pose_3d = 1
        except KeyError:
            self.has_pose_3d = 0
        if ignore_3d or dataset == "mpi-inf-3dhp":
            self.has_pose_3d = 0
        
        # Get 2D keypoints
        try:
            keypoints_gt = self.data['part']
        except KeyError:
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
        try:
            keypoints_openpose = self.data['openpose']
        except KeyError:
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))
        
        if dataset == "mpi-inf-3dhp" or dataset == "h36m":
            keypoints_gt = np.zeros((len(self.imgname), 24, 3))
            keypoints_openpose = np.zeros((len(self.imgname), 25, 3))

        self.keypoints = np.concatenate([keypoints_openpose, keypoints_gt], axis=1)

        # Get gender data, if available
        try:
            gender = self.data['gender']
            self.gender = np.array([0 if str(g) == 'm' else 1 for g in gender]).astype(np.int32)
        except KeyError:
            self.gender = -1*np.ones(len(self.imgname)).astype(np.int32)
        
        self.length = self.scale.shape[0]
        
        # Get person id for mask image
        if self.is_train:
            self.p_id = self.data['p_id']

        # Get has rendering
        if self.is_train:
            try:
                self.has_rendering = self.data['has_rendering']
            except KeyError:
                self.has_rendering = np.zeros(len(self.imgname))

        # For Jigsaw Puzzle Input
        self.num_patch = 4
        self.divider = 2
        self.list = list(range(4))
        self.permutations = np.array(list(itertools.permutations(self.list, len(self.list))))

        self.augment_tile_256_224 = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        self.transform_224 = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
        ])

        self.transform_224_tensor = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])

        # For Rotation Input
        self.angle_list = [0, 90, 180, 270]

    def augm_params(self):
        """Get augmentation parameters."""
        flip = 0            # flipping
        pn = np.ones(3)  # per channel pixel-noise
        rot = 0            # rotation
        sc = 1            # scaling
        if self.is_train:
            # We flip with probability 1/2
            if np.random.uniform() <= 0.5:
                flip = 1
            
            # Each channel is multiplied with a number 
            # in the area [1-opt.noiseFactor,1+opt.noiseFactor]
            pn = np.random.uniform(1-self.options.noise_factor, 1+self.options.noise_factor, 3)
            
            # The rotation is a number in the area [-2*rotFactor, 2*rotFactor]
            rot = min(2*self.options.rot_factor,
                    max(-2*self.options.rot_factor, np.random.randn()*self.options.rot_factor))
            
            # The scale is multiplied with a number
            # in the area [1-scaleFactor,1+scaleFactor]
            sc = min(1+self.options.scale_factor,
                    max(1-self.options.scale_factor, np.random.randn()*self.options.scale_factor+1))
            # but it is zero with probability 3/5
            if np.random.uniform() <= 0.6:
                rot = 0
        
        return flip, pn, rot, sc

    def rgb_processing(self, rgb_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        rgb_img = crop(rgb_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            rgb_img = flip_img(rgb_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        rgb_img[:,:,0] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,0]*pn[0]))
        rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        img_for_3task = np.uint8(copy.deepcopy(rgb_img))
        rgb_img = np.transpose(rgb_img.astype('float32'),(2,0,1))/255.0
        return rgb_img, img_for_3task

    def mask_processing(self, mask_img, center, scale, rot, flip, pn):
        """Process rgb image and do augmentation."""
        mask_img = crop(mask_img, center, scale, 
                      [constants.IMG_RES, constants.IMG_RES], rot=rot)
        # flip the image 
        if flip:
            mask_img = flip_img(mask_img)
        # in the rgb image we add pixel noise in a channel-wise manner
        mask_img[:,:] = np.minimum(255.0, np.maximum(0.0, mask_img[:,:]*pn[0]))
        # rgb_img[:,:,1] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,1]*pn[1]))
        # rgb_img[:,:,2] = np.minimum(255.0, np.maximum(0.0, rgb_img[:,:,2]*pn[2]))
        # (3,224,224),float,[0,1]
        mask_img = mask_img.astype('float32')/255.0
        return mask_img

    def j2d_processing(self, kp, center, scale, r, f):
        """Process gt 2D keypoints and apply all augmentation transforms."""
        nparts = kp.shape[0]
        for i in range(nparts):
            kp[i,0:2] = transform(kp[i,0:2]+1, center, scale, 
                                  [constants.IMG_RES, constants.IMG_RES], rot=r)
        # convert to normalized coordinates
        kp[:,:-1] = 2.*kp[:,:-1]/constants.IMG_RES - 1.
        # flip the x coordinates
        if f:
             kp = flip_kp(kp)
        kp = kp.astype('float32')
        return kp

    def j3d_processing(self, S, r, f):
        """Process gt 3D keypoints and apply all augmentation transforms."""
        # in-plane rotation
        rot_mat = np.eye(3)
        if not r == 0:
            rot_rad = -r * np.pi / 180
            sn,cs = np.sin(rot_rad), np.cos(rot_rad)
            rot_mat[0,:2] = [cs, -sn]
            rot_mat[1,:2] = [sn, cs]
        
        S[:, :-1] = np.einsum('ij,kj->ki', rot_mat, S[:, :-1]) 
        # flip the x coordinates
        if f:
            S = flip_kp(S)
        S = S.astype('float32')
        return S

    def pose_processing(self, pose, r, f):
        """Process SMPL theta parameters  and apply all augmentation transforms."""
        # rotation or the pose parameters
        pose[:3] = rot_aa(pose[:3], r)
        # flip the pose parameters
        if f:
            pose = flip_pose(pose)
        # (72),float
        pose = pose.astype('float32')
        return pose

    def __getitem__(self, index):
        item = {}
        scale = self.scale[index].copy()
        center = self.center[index].copy()

        # Get augmentation parameters
        flip, pn, rot, sc = self.augm_params()
        
        # Load image
        imgname = join(self.img_dir, self.imgname[index])

        try:
            img = cv2.imread(imgname)[:,:,::-1].copy().astype(np.float32)
        except TypeError:
            print(imgname)
        orig_shape = np.array(img.shape)[:2]

        # Get SMPL parameters, if available
        if self.has_smpl[index]:
            pose = self.pose[index].copy()
            betas = self.betas[index].copy()
        else:
            pose = np.zeros(72)
            betas = np.zeros(10)

        # Process image
        img, img_for_3task = self.rgb_processing(img, center, sc*scale, rot, flip, pn)
        img_for_3task = Image.fromarray(img_for_3task)
        img = torch.from_numpy(img).float()

        if self.is_train:
            # Process Context Encoder Input
            context_encoder_input = copy.deepcopy(img)
            center_crop_img = copy.deepcopy(context_encoder_input[:, 80:144, 80:144])
            context_encoder_input[:, 80:144, 80:144] = 0

            # Process Jigsaw Puzzle Input
            image_224 = self.transform_224(copy.deepcopy(img_for_3task))
            W, H = image_224.size # W, H 224
            
            s = W/self.divider  # 112
            a = s/2 # 56

            tiles_jigsaw = [None] * self.num_patch
            for n in range(self.num_patch):
                i = n//self.divider
                j = n%self.divider
                c = [a * i * 2 + a, a * j * 2 + a]
                c = np.array([c[1] - a, c[0] - a, c[1] + a + 1, c[0] + a + 1]).astype(int)

                tile = image_224.crop(c.tolist())
                tile_jigsaw = self.augment_tile_256_224(tile)
                tiles_jigsaw[n] = tile_jigsaw

            jigsaw_order = np.random.randint(len(self.permutations))
            jigsaw_input = [tiles_jigsaw[self.permutations[jigsaw_order][t]] for t in range(self.num_patch)]
            jigsaw_input = torch.stack(jigsaw_input, 0)
            
            # Process Roation Net Input
            rotation_idx = np.random.randint(4)
            angle = self.angle_list[rotation_idx]
            rotation_input = self.rotate_img(copy.deepcopy(img_for_3task), angle)
            rotation_input = self.transform_224_tensor(rotation_input)
        
        if self.is_train:
            # Load mask image
            p_id = self.p_id[index]
            if self.dataset in ["h36m", "lsp-orig", "mpii"]:
                mask_imgpath = imgname.replace("/images/", "/masks/")
            elif self.dataset == "youtube":
                mask_imgpath = imgname.replace("/imageFiles/", "/masks/")
            elif self.dataset == "coco":
                mask_imgpath = imgname.replace("/train2014/", "/masks/")    
            elif self.dataset == "mpi-inf-3dhp":
                mask_imgpath = imgname.replace("/imageFrames/", "/masks/")
            elif self.dataset == "new_data":
                mask_imgpath = imgname.replace("/imageFrames/", "/masks/")

            mask_imgname_dir = osp.dirname(mask_imgpath)
            mask_imgname = osp.basename(mask_imgpath)
            mask_imgname = str(p_id) + "_" + mask_imgname
            mask_imgname = join(mask_imgname_dir, mask_imgname)

            try:
                mask_img = cv2.imread(mask_imgname, 0)[:,:].copy().astype(np.float32)
                has_mask_gt = 1
                if np.sum(mask_img) == 0:
                    raise TypeError
                
                # Process mask image
                mask_img = self.mask_processing(mask_img, center, sc*scale, rot, flip, pn)
                mask_img = torch.from_numpy(mask_img).float()
            except TypeError:
                mask_img = np.zeros((224, 224)).astype(np.float32)
                mask_img = torch.from_numpy(mask_img).float()
                has_mask_gt = 0

        if self.is_train:
            # Store image for 3 Task
            item['context_encoder_input'] = self.normalize_img(context_encoder_input)
            item['center_crop_img'] = self.normalize_img(center_crop_img)
            item['jigsaw_input'] = jigsaw_input
            item['rotation_input'] = rotation_input
            item['jigsaw_order'] = jigsaw_order
            item['rotation_idx'] = rotation_idx

        # Store image before normalization to use it in visualization
        item['img'] = self.normalize_img(img)
        item['pose'] = torch.from_numpy(self.pose_processing(pose, rot, flip)).float()
        item['betas'] = torch.from_numpy(betas).float()
        item['imgname'] = imgname
        item['black_img'] = self.normalize_img(torch.from_numpy(np.zeros((3, 224, 224))))

        if self.is_train:
            # Store mask image
            item['gt_mask'] = mask_img
            item['has_mask'] = has_mask_gt
        
        # Get 3D pose, if available
        if self.has_pose_3d:
            S = self.pose_3d[index].copy()
            item['pose_3d'] = torch.from_numpy(self.j3d_processing(S, rot, flip)).float()
        else:
            item['pose_3d'] = torch.zeros(24,4, dtype=torch.float32)

        # Get 2D keypoints and apply augmentation transforms
        keypoints = self.keypoints[index].copy()
        item['keypoints'] = torch.from_numpy(self.j2d_processing(keypoints, center, sc*scale, rot, flip)).float()
        item['has_smpl'] = self.has_smpl[index]
        item['has_pose_3d'] = self.has_pose_3d
        item['scale'] = float(sc * scale)
        item['center'] = center.astype(np.float32)
        item['orig_shape'] = orig_shape
        item['is_flipped'] = flip
        item['rot_angle'] = np.float32(rot)
        item['gender'] = self.gender[index]
        item['sample_index'] = index
        item['dataset_name'] = self.dataset
        if self.is_train:
            item['p_id'] = p_id
            item["has_rendering"] = self.has_rendering[index]

        try:
            item['maskname'] = self.maskname[index]
        except AttributeError:
            item['maskname'] = ''
        try:
            item['partname'] = self.partname[index]
        except AttributeError:
            item['partname'] = ''

        return item

    def __len__(self):
        return len(self.imgname)

    def rotate_img(self, img, rot):
        img = np.array(img)
        if rot == 0: # 0 degrees rotation
            img = img
        elif rot == 90: # 90 degrees rotation
            img = np.flipud(np.transpose(img, (1,0,2)))
        elif rot == 180: # 180 degrees rotation
            img = np.fliplr(np.flipud(img))
        elif rot == 270: # 270 degrees rotation / or -90
            img = np.transpose(np.flipud(img), (1,0,2))
        else:
            raise ValueError('rotation should be 0, 90, 180, or 270 degrees')

        return Image.fromarray(img)
