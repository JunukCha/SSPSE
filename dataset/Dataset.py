import glob, time
import numpy as np
from PIL import Image
import os
import os.path as osp
import cv2

import torch
from torchvision import transforms
from torchvision.transforms import Normalize
import torchvision
import itertools
import copy
import joblib
from pymatreader import read_mat

from lib.data_utils.img_utils import normalize_2d_kp, transfrom_keypoints, get_single_image_crop
from dataset.data_utils import (
    xyz2uvd,
    kp_to_bbox_param,
    get_dataset_path,
    crop,
    transform
)


class SPINDataset():
    def __init__(self, dataset):
        self.dataset = dataset
        self.data = joblib.load("/data/3dpw/3dpw_train_db.pt")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        feature = self.data["features"][idx]
        joints3d = self.data["joints3D"][idx]
        joints2d = self.data["joints2D"][idx]
        shape = self.data["shape"][idx]
        pose = self.data["pose"][idx]
        return feature, joints3d, joints2d, shape, pose


class TrainDataset():
    def __init__(self, train_data, use_seg_gt=False, joints_dataset="3dpw", train_texture_net=False, use_augmentation=False, use_coco_gt=False):
        self.num_patch = 4
        self.divider = 2
        self.use_seg_gt = use_seg_gt
        self.joints_dataset = joints_dataset
        self.use_augmentation = use_augmentation

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
        
        self.angle_list = [0, 90, 180, 270]

        if train_texture_net:
            h36m_train_data_path = "/data/human3.6m/h36m_train_db_simple.pt"
            self.h36m_train_dataset = self.load_dataset(h36m_train_data_path, prop_sample=0.08) # 5013
            print("H36M Train dataset: {}".format(format(self.h36m_train_dataset["len_data"], ",")))

            # three_dp_train_data_path = "/data/3dpw/3dpw_train_db_simple.pt"
            # self.three_dp_train_dataset = self.load_dataset(three_dp_train_data_path, prop_sample=0.2)
            # print("3DPW Train dataset: {}".format(format(self.three_dp_train_dataset["len_data"], ",")))
            
            mpi_inf_train_data_path = "/data/MPI-INF-3D-HP/mpi-inf-3d-hp_train_db_simple.pt"
            self.mpi_inf_train_dataset = self.load_dataset(mpi_inf_train_data_path, prop_sample=0.05) # 4835
            print("MPI-INF Train dataset: {}".format(format(self.mpi_inf_train_dataset["len_data"], ",")))

            mpii_train_data_path = "/data/MPII/mpii_train_db_simple.pt"
            self.mpii_train_dataset = self.load_dataset(mpii_train_data_path, prop_sample=1) # 5445
            print("MPII Train dataset: {}".format(format(self.mpii_train_dataset["len_data"], ",")))

            coco_train_data_path = "/data/coco/coco_train_db_simple.pt" 
            self.coco_train_dataset = self.load_dataset(coco_train_data_path, prop_sample=1) # 4040
            print("COCO Train dataset: {}".format(format(self.coco_train_dataset["len_data"], ",")))

            youtube_train_data_path = "/data/youtube_collection/youtube_collection_train_db_simple.pt"
            self.youtube_train_dataset = self.load_dataset(youtube_train_data_path, prop_sample=1) # 3045
            print("Youtube Train dataset: {}".format(format(self.youtube_train_dataset["len_data"], ",")))

            key_list = ["p_id_list", "img_path_list", "bbox_list",
                            "joints3D_list", "has_joints3D", "joints2D_list",
                            "has_joints2D", "shape_list", "has_shape",
                            "pose_list", "has_pose", "has_camera"]
            dataset_dict = {}
            for key in key_list:
                dataset_dict[key] =  np.concatenate((
                                self.h36m_train_dataset[key], 
                                self.mpi_inf_train_dataset[key],
                                self.mpii_train_dataset[key],
                                self.coco_train_dataset[key], 
                                self.youtube_train_dataset[key],))
                
            self.p_id_list = dataset_dict["p_id_list"]
            self.img_path_list = dataset_dict["img_path_list"]
            self.bbox_list = dataset_dict["bbox_list"]
            self.joints3D_list = dataset_dict["joints3D_list"]
            self.has_joints3D = dataset_dict["has_joints3D"]
            self.joints2D_list = dataset_dict["joints2D_list"]
            self.has_joints2D = dataset_dict["has_joints2D"]
            self.shape_list = dataset_dict["shape_list"]
            self.has_shape = dataset_dict["has_shape"]
            self.pose_list = dataset_dict["pose_list"]
            self.has_pose = dataset_dict["has_pose"]
            self.has_camera = dataset_dict["has_camera"]
            del dataset_dict
            del self.h36m_train_dataset
            del self.mpi_inf_train_dataset
            del self.mpii_train_dataset
            del self.coco_train_dataset
            del self.youtube_train_dataset

        else:
            if train_data == 0:
                if self.use_seg_gt:
                    train_dataset_path = "/data/human3.6m/h36m_train_db_simple_seg.pt"
                else:
                    train_dataset_path = "/data/human3.6m/h36m_train_db_simple.pt"

                self.train_dataset = self.load_dataset(train_dataset_path, use_gt=True)
                print("{} Train dataset: {}".format(joints_dataset.upper(), format(self.train_dataset["len_data"], ",")))

                if self.use_seg_gt:
                    test_dataset_path = "/data/human3.6m/h36m_test_db_simple_seg.pt"
                else:
                    test_dataset_path = "/data/human3.6m/h36m_test_db_simple.pt"

                self.test_dataset = self.load_dataset(test_dataset_path, use_gt=True)
                print("{} Train dataset: {}".format(joints_dataset.upper(), format(self.test_dataset["len_data"], ",")))

            if train_data in [1, 2, 3, 4, 5]:
                ### 3DPW ###
                if joints_dataset == "3dpw":
                    dataset_path = "/data/3dpw/3dpw_train_db_simple.pt"
                elif joints_dataset == "h36m":
                    if self.use_seg_gt:
                        dataset_path = "/data/human3.6m/h36m_train_db_simple_seg.pt"
                    else:
                        dataset_path = "/data/human3.6m/h36m_train_db_simple.pt"
                self.dataset = self.load_dataset(dataset_path, use_gt=True)
                print("{} Train dataset: {}".format(joints_dataset.upper(), format(self.dataset["len_data"], ",")))

            if train_data in [2, 4]:
                ### Human 3.6M ###
                if joints_dataset == "3dpw":
                    dataset2_path = "/data/human3.6m/h36m_train_db_simple.pt"
                    self.dataset2 = self.load_dataset(dataset2_path)
                    print("H36M Train dataset: {}".format(format(self.dataset2["len_data"], ",")))
                elif joints_dataset == "h36m":
                    dataset2_path = "/data/3dpw/3dpw_train_db_simple.pt"
                    self.dataset2 = self.load_dataset(dataset2_path)
                    print("3DPW Train dataset: {}".format(format(self.dataset2["len_data"], ",")))
                
                ### MPII ###
                mpii_dataset_path = "/data/MPI-INF-3D-HP/mpi-inf-3d-hp_train_db_simple.pt"
                self.mpii_dataset = self.load_dataset(mpii_dataset_path)
                print("MPII Train dataset: {}".format(format(self.mpii_dataset["len_data"], ",")))

                ### SYSU ###
                sysu_dataset_path = "/data/SYSU3DAction/sysu_train_db_simple.pt"
                self.sysu_dataset = self.load_dataset(sysu_dataset_path)
                print("SYSU Train dataset: {}".format(format(self.sysu_dataset["len_data"], ",")))

                ### Nucla ###
                nucla_dataset_path = "/data/nucla/nucla_train_db_simple.pt"
                self.nucla_dataset = self.load_dataset(nucla_dataset_path)
                print("Nucla Train dataset: {}".format(format(self.nucla_dataset["len_data"], ",")))
            
            if train_data in [3, 4]:
                ### ETH ###
                eth_dataset_path = "/data/ETH_padding/ETH_train_db_simple.pt"
                self.eth_dataset = self.load_dataset(eth_dataset_path)
                print("ETH Train dataset: {}".format(format(self.eth_dataset["len_data"], ",")))
                
                ### Penn Fudan ###
                penn_fudan_dataset_path = "/data/Penn-Fudan/Penn-Fudan_train_db_simple.pt"
                self.penn_fudan_dataset = self.load_dataset(penn_fudan_dataset_path)
                print("Penn-Fudan Train dataset: {}".format(format(self.penn_fudan_dataset["len_data"], ",")))

            if train_data == 5:
                if self.use_seg_gt:
                    coco_dataset_path = "/data/coco/coco_train_db_simple_seg.pt"
                else:
                    coco_dataset_path = "/data/coco/coco_train_db_simple.pt"
                if use_coco_gt:
                    self.coco_dataset = self.load_dataset(coco_dataset_path, use_gt=True)
                else:
                    self.coco_dataset = self.load_dataset(coco_dataset_path)
                print("COCO Train dataset: {}".format(format(self.coco_dataset["len_data"], ",")))

            if train_data == 0:
                key_list = ["p_id_list", "img_path_list", "bbox_list",
                            "joints3D_list", "has_joints3D", "joints2D_list",
                            "has_joints2D", "shape_list", "has_shape",
                            "pose_list", "has_pose", "has_camera"]
                dataset_dict = {}
                for key in key_list:
                    dataset_dict[key] =  np.concatenate((
                                    self.train_dataset[key], 
                                    self.test_dataset[key]))
                    
                self.p_id_list = dataset_dict["p_id_list"]
                self.img_path_list = dataset_dict["img_path_list"]
                self.bbox_list = dataset_dict["bbox_list"]
                self.joints3D_list = dataset_dict["joints3D_list"]
                self.has_joints3D = dataset_dict["has_joints3D"]
                self.joints2D_list = dataset_dict["joints2D_list"]
                self.has_joints2D = dataset_dict["has_joints2D"]
                self.shape_list = dataset_dict["shape_list"]
                self.has_shape = dataset_dict["has_shape"]
                self.pose_list = dataset_dict["pose_list"]
                self.has_pose = dataset_dict["has_pose"]
                self.has_camera = dataset_dict["has_camera"]
                del dataset_dict
                del self.train_dataset
                del self.test_dataset

            if train_data == 1:
                self.p_id_list = self.dataset["p_id_list"]
                self.img_path_list = self.dataset["img_path_list"]
                self.bbox_list = self.dataset["bbox_list"]
                self.joints3D_list = self.dataset["joints3D_list"]
                self.has_joints3D = self.dataset["has_joints3D"]
                self.joints2D_list = self.dataset["joints2D_list"]
                self.has_joints2D = self.dataset["has_joints2D"]
                self.shape_list = self.dataset["shape_list"]
                self.has_shape = self.dataset["has_shape"]
                self.pose_list = self.dataset["pose_list"]
                self.has_pose = self.dataset["has_pose"]
                self.has_camera = self.dataset["has_camera"]
                del self.dataset
            
            if train_data == 2:
                key_list = ["p_id_list", "img_path_list", "bbox_list",
                            "joints3D_list", "has_joints3D", "joints2D_list",
                            "has_joints2D", "shape_list", "has_shape",
                            "pose_list", "has_pose", "has_camera"]
                dataset_dict = {}
                for key in key_list:
                    dataset_dict[key] =  np.concatenate((
                                    self.dataset[key], 
                                    self.dataset2[key],
                                    self.mpii_dataset[key],
                                    self.sysu_dataset[key], 
                                    self.nucla_dataset[key]))
                    
                self.p_id_list = dataset_dict["p_id_list"]
                self.img_path_list = dataset_dict["img_path_list"]
                self.bbox_list = dataset_dict["bbox_list"]
                self.joints3D_list = dataset_dict["joints3D_list"]
                self.has_joints3D = dataset_dict["has_joints3D"]
                self.joints2D_list = dataset_dict["joints2D_list"]
                self.has_joints2D = dataset_dict["has_joints2D"]
                self.shape_list = dataset_dict["shape_list"]
                self.has_shape = dataset_dict["has_shape"]
                self.pose_list = dataset_dict["pose_list"]
                self.has_pose = dataset_dict["has_pose"]
                self.has_camera = dataset_dict["has_camera"]
                del dataset_dict
                del self.dataset
                del self.dataset2
                del self.mpii_dataset
                del self.sysu_dataset
                del self.nucla_dataset

            if train_data == 3:
                key_list = ["p_id_list", "img_path_list", "bbox_list",
                            "joints3D_list", "has_joints3D", "joints2D_list",
                            "has_joints2D", "shape_list", "has_shape",
                            "pose_list", "has_pose", "has_camera"]
                dataset_dict = {}
                for key in key_list:
                    dataset_dict[key] =  np.concatenate((
                                    self.dataset[key], 
                                    self.eth_dataset[key],
                                    self.penn_fudan_dataset[key]))

                    
                self.p_id_list = dataset_dict["p_id_list"]
                self.img_path_list = dataset_dict["img_path_list"]
                self.bbox_list = dataset_dict["bbox_list"]
                self.joints3D_list = dataset_dict["joints3D_list"]
                self.has_joints3D = dataset_dict["has_joints3D"]
                self.joints2D_list = dataset_dict["joints2D_list"]
                self.has_joints2D = dataset_dict["has_joints2D"]
                self.shape_list = dataset_dict["shape_list"]
                self.has_shape = dataset_dict["has_shape"]
                self.pose_list = dataset_dict["pose_list"]
                self.has_pose = dataset_dict["has_pose"]
                self.has_camera = dataset_dict["has_camera"]
                del dataset_dict
                del self.dataset
                del self.eth_dataset
                del self.penn_fudan_dataset

            if train_data == 4:
                key_list = ["p_id_list", "img_path_list", "bbox_list",
                            "joints3D_list", "has_joints3D", "joints2D_list",
                            "has_joints2D", "shape_list", "has_shape",
                            "pose_list", "has_pose", "has_camera"]
                dataset_dict = {}
                for key in key_list:
                    dataset_dict[key] =  np.concatenate((
                                    self.dataset[key], 
                                    self.dataset2[key],
                                    self.mpii_dataset[key],
                                    self.sysu_dataset[key], 
                                    self.nucla_dataset[key],
                                    self.eth_dataset[key],
                                    self.penn_fudan_dataset[key]))

                    
                self.p_id_list = dataset_dict["p_id_list"]
                self.img_path_list = dataset_dict["img_path_list"]
                self.bbox_list = dataset_dict["bbox_list"]
                self.joints3D_list = dataset_dict["joints3D_list"]
                self.has_joints3D = dataset_dict["has_joints3D"]
                self.joints2D_list = dataset_dict["joints2D_list"]
                self.has_joints2D = dataset_dict["has_joints2D"]
                self.shape_list = dataset_dict["shape_list"]
                self.has_shape = dataset_dict["has_shape"]
                self.pose_list = dataset_dict["pose_list"]
                self.has_pose = dataset_dict["has_pose"]
                self.has_camera = dataset_dict["has_camera"]
                del dataset_dict
                del self.dataset
                del self.dataset2
                del self.mpii_dataset
                del self.sysu_dataset
                del self.nucla_dataset
                del self.eth_dataset
                del self.penn_fudan_dataset

            if train_data == 5:
                key_list = ["p_id_list", "img_path_list", "bbox_list",
                            "joints3D_list", "has_joints3D", "joints2D_list",
                            "has_joints2D", "shape_list", "has_shape",
                            "pose_list", "has_pose", "has_camera"]
                dataset_dict = {}
                for key in key_list:
                    dataset_dict[key] =  np.concatenate((
                                    self.dataset[key], 
                                    self.coco_dataset[key]))

                self.p_id_list = dataset_dict["p_id_list"]
                self.img_path_list = dataset_dict["img_path_list"]
                self.bbox_list = dataset_dict["bbox_list"]
                self.joints3D_list = dataset_dict["joints3D_list"]
                self.has_joints3D = dataset_dict["has_joints3D"]
                self.joints2D_list = dataset_dict["joints2D_list"]
                self.has_joints2D = dataset_dict["has_joints2D"]
                self.shape_list = dataset_dict["shape_list"]
                self.has_shape = dataset_dict["has_shape"]
                self.pose_list = dataset_dict["pose_list"]
                self.has_pose = dataset_dict["has_pose"]
                self.has_camera = dataset_dict["has_camera"]
                del dataset_dict
                del self.dataset
                del self.coco_dataset

        self.scale = 1.2
        print("Total Train dataset: {}".format(format(self.__len__(), ",")))

    def load_dataset(self, dataset_path, use_gt=False, prop_sample=None):
        dataset_dict = {}
        data = joblib.load(dataset_path)
        p_id_list = np.array(data["p_id"])
        img_path_list = np.array(data["img_name"])
        bbox_list = np.array(data["bbox"])
        len_data = len(img_path_list)
        
        J24_TO_J17 = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 18, 14, 16, 17]
        J24_TO_J14 = J24_TO_J17[:14]

        UNIVERSAL_BODIES = [
            16,  # R ankle
            14,  # R knee
            12,  # R hip
            11,  # L hip
            13,  # L knee
            15,  # L ankle
            10,  # R Wrist
            8,  # R Elbow
            6,  # R shoulder
            5,  # L shoulder
            7,  # L Elbow
            9,  # L Wrist
            0,
            0,
        ]
        
        if use_gt:
            try:
                joints3D_list = data["joints3D"]
                joints3D_list = np.array(joints3D_list)
                if self.joints_dataset == "h36m":
                    joints3D_list = joints3D_list[:, :, :-1]
                has_joints3D = np.ones((len_data, 14, 1))
            except KeyError:
                print("Has No 3D joints")
                if self.joints_dataset == "h36m":
                    joints3D_list = np.zeros((len_data, 24, 3))
                else:
                    joints3D_list = np.zeros((len_data, 49, 3))
                has_joints3D = np.zeros((len_data, 14, 1))
            
            try:
                joints2D_list = data["joints2D"]
                joints2D_list = np.array(joints2D_list)
                if self.joints_dataset == "h36m":
                    if joints2D_list.shape[1] == 24:
                        joints2D_list = joints2D_list[:, J24_TO_J14, :]
                    elif joints2D_list.shape[1] == 17:
                        joints2D_list = joints2D_list[:, UNIVERSAL_BODIES, :]
                        joints2D_list[: :13, 2] = 1
                        joints2D_list[:, 13, 2] = 0
                        joints2D_list[:, 14, 2] = 0
                has_joints2D = np.ones((len_data, 14, 1))
            except KeyError:
                print("Has No 2D joints")
                joints2D_list = np.zeros((len_data, 14, 3))
                has_joints2D = np.zeros((len_data, 14, 1))
            
            try:
                shape_list = data["shape"]
                has_shape = np.ones((len_data, 1))
            except KeyError:
                print("Has No Shape")
                shape_list = np.zeros((len_data, 10))
                has_shape = np.zeros((len_data, 1))
            
            try:
                pose_list = data["pose"]
                has_pose = np.ones((len_data, 24, 3, 1))
            except KeyError:
                print("Has No Pose")
                pose_list = np.zeros((len_data, 72))
                has_pose = np.zeros((len_data, 24, 3, 1))
        
            has_camera = np.ones((len_data, 1))
        else:
            if self.joints_dataset == "h36m":
                joints3D_list = np.zeros((len_data, 24, 3))
            else:
                joints3D_list = np.zeros((len_data, 49, 3))
            has_joints3D = np.zeros((len_data, 14, 1))
            
            joints2D_list = np.zeros((len_data, 14, 3))
            has_joints2D = np.zeros((len_data, 14, 1))

            shape_list = np.zeros((len_data, 10))
            has_shape = np.zeros((len_data, 1))

            pose_list = np.zeros((len_data, 72))
            has_pose = np.zeros((len_data, 24, 3, 1))

            has_camera = np.zeros((len_data, 1))

        if prop_sample:
            np.random.seed(0)
            random_idx = np.random.choice(range(len_data), int(len_data*prop_sample))
            dataset_dict["p_id_list"] = p_id_list[random_idx]
            dataset_dict["img_path_list"] = img_path_list[random_idx]
            print(np.random.choice(dataset_dict["img_path_list"], 10))
            dataset_dict["bbox_list"] = bbox_list[random_idx]
            dataset_dict["len_data"] = int(len_data*prop_sample)
            dataset_dict["joints3D_list"] = joints3D_list[random_idx]
            dataset_dict["has_joints3D"] = has_joints3D[random_idx]
            dataset_dict["joints2D_list"] = joints2D_list[random_idx]
            dataset_dict["has_joints2D"] = has_joints2D[random_idx]
            dataset_dict["shape_list"] = shape_list[random_idx]
            dataset_dict["has_shape"] = has_shape[random_idx]
            dataset_dict["pose_list"] = pose_list[random_idx]
            dataset_dict["has_pose"] = has_pose[random_idx]
            dataset_dict["has_camera"] = has_camera[random_idx]
        else:
            dataset_dict["p_id_list"] = p_id_list
            dataset_dict["img_path_list"] = img_path_list
            dataset_dict["bbox_list"] = bbox_list
            dataset_dict["len_data"] = len_data
            dataset_dict["joints3D_list"] = joints3D_list
            dataset_dict["has_joints3D"] = has_joints3D
            dataset_dict["joints2D_list"] = joints2D_list
            dataset_dict["has_joints2D"] = has_joints2D
            dataset_dict["shape_list"] = shape_list
            dataset_dict["has_shape"] = has_shape
            dataset_dict["pose_list"] = pose_list
            dataset_dict["has_pose"] = has_pose
            dataset_dict["has_camera"] = has_camera
        return dataset_dict


    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        p_id = int(self.p_id_list[idx])
        img_path = self.img_path_list[idx]
        bbox = self.bbox_list[idx]

        if self.use_seg_gt:
            seg_img_path = img_path.split("/")
            seg_img_path[3] = "segFiles"
            seg_img_path[-1] = str(p_id) + "_" + seg_img_path[-1]
            seg_img_path = "/".join(seg_img_path)
            _, seg_img = get_single_image_crop(seg_img_path, bbox, self.scale)
        
        image, _ = get_single_image_crop(img_path, bbox, self.scale)
        image = Image.fromarray(image)
        img = self.transform_224_tensor(image)
        image_224 = self.transform_224(image)
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

        # Context Encoder
        context_encoder_input = copy.deepcopy(img)
        center_crop_img = copy.deepcopy(context_encoder_input[:, 80:144, 80:144])
        context_encoder_input[:, 80:144, 80:144] = 0

        # Jigsaw Puzzle
        jigsaw_order = np.random.randint(len(self.permutations))
        jigsaw_input = [tiles_jigsaw[self.permutations[jigsaw_order][t]] for t in range(self.num_patch)]
        jigsaw_input = torch.stack(jigsaw_input, 0)
        
        # Rotation
        rotation_idx = np.random.randint(4)
        angle = self.angle_list[rotation_idx]
        rotation_input = self.rotate_img(image, angle)
        rotation_input = self.transform_224_tensor(rotation_input)

        joints3d = self.joints3D_list[idx]
        joints3d = torch.FloatTensor(joints3d)
        has_joints3d = self.has_joints3D[idx]
        has_joints3d = torch.FloatTensor(has_joints3d)

        joints2d = self.joints2D_list[idx]
        joints2d[:,:2], trans = transfrom_keypoints(
            kp_2d=joints2d[:,:2],
            center_x=bbox[0],
            center_y=bbox[1],
            width=bbox[2],
            height=bbox[3],
            patch_width=224,
            patch_height=224,
            do_augment=False,
        )
        joints2d[:,:2] = normalize_2d_kp(joints2d[:,:2], 224)
        joints2d = torch.FloatTensor(joints2d)
        has_joints2d = self.has_joints2D[idx]
        has_joints2d = torch.FloatTensor(has_joints2d)
        
        shape = self.shape_list[idx]
        shape = torch.FloatTensor(shape)
        has_shape = self.has_shape[idx]
        has_shape = torch.FloatTensor(has_shape)

        pose = self.pose_list[idx]
        pose = torch.FloatTensor(pose)
        has_pose = self.has_pose[idx]
        has_pose = torch.FloatTensor(has_pose)

        has_camera = self.has_camera[idx]
        has_camera = torch.FloatTensor(has_camera)
        
        item = {}
        item['img_path'] = img_path
        item['p_id'] = p_id
        item['img'] = img
        item['context_encoder_input'] = context_encoder_input
        item['center_crop_img'] = center_crop_img
        item['jigsaw_input'] = jigsaw_input
        item['rotation_input'] = rotation_input
        item['jigsaw_order'] = jigsaw_order
        item['rotation_idx'] = rotation_idx
        item['joints3d'] = joints3d
        item['has_joints3d'] = has_joints3d
        item['joints2d'] = joints2d
        item['has_joints2d'] = has_joints2d
        item['shape'] = shape
        item['has_shape'] = has_shape
        item['pose'] = pose
        item['has_pose'] = has_pose
        item['has_camera'] = has_camera
        if self.use_seg_gt:
            item['target_seg'] = seg_img[:1, :, :]
        return item

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


class TestDataset():
    def __init__(self, dataset, train=False):
        # self.normalize_img = Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        self.transform_224_tensor = transforms.Compose([
            transforms.Resize((224, 224), Image.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self.dataset = dataset
        if self.dataset == "3dpw":
            self.data = joblib.load("/data/3dpw/3dpw_test_db_m_f.pt")
            # self.crop_img_path_list = self.data["crop_img_name"]
            self.img_path_list = self.data["img_name"]
            self.bbox_list = self.data["bbox"]
            self.joints3d_list = self.data["joints3D"]
            self.scale = 1.2
            print("{} Test dataset: {}".format(self.dataset.upper(), format(self.__len__(), ",")))

        elif self.dataset == "h36m":
            # self.data = joblib.load("/data/human3.6m/h36m_eval_db.pt")
            # self.crop_img_path_list = self.data["crop_img_name"]
            # self.data = np.load("data/npz_file/h36m_valid_protocol2.npz")
            if train:
                self.data = joblib.load("/data/human3.6m/h36m_train_db_simple_seg.pt")
            else:
                self.data = joblib.load("/data/human3.6m/h36m_test_db.pt")

            self.img_path_list = self.data["img_name"]
            self.bbox_list = self.data["bbox"]
            self.joints3d_list = self.data["joints3D"]
            self.scale = 1.2
            print("{} Test dataset: {}".format(self.dataset.upper(), format(self.__len__(), ",")))

        elif self.dataset == "lsp":
            # self.data = joblib.load("/data/human3.6m/h36m_eval_db.pt")
            # self.crop_img_path_list = self.data["crop_img_name"]
            self.data = joblib.load("/data/lsp/lsp_test_db.pt")
            # self.data = joblib.load("/data/MPI-INF-3D-HP/mpi_test_db.pt")
            self.img_path_list = self.data["img_name"]
            self.bbox_list = self.data["bbox"]
            self.gt_parts_path = self.data["gt_parts_name"]
            self.gt_seg_path = self.data["gt_seg_name"]
            self.center = self.data["center"]
            self.orig_shape = self.data["orig_shape"]
            self.scale = 1.2
            print("{} Test dataset: {}".format(self.dataset.upper(), format(self.__len__(), ",")))

        else:
            raise Exception("There is no test data!")

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, idx):
        # crop_img_path = self.crop_img_path_list[idx]
        # crop_img = Image.open(crop_img_path)
        # crop_img = self.transform_224_tensor(crop_img)

        img_path = self.img_path_list[idx]

        if self.dataset == "h36m":
            img_path = osp.join("/data/human3.6m", img_path)
        bbox = self.bbox_list[idx]
        _, crop_img = get_single_image_crop(img_path, bbox, self.scale)
        item = {}

        if self.dataset == "h36m" or self.dataset == "3dpw":
            pose_3d = self.joints3d_list[idx]
            pose_3d = torch.from_numpy(pose_3d).float()
            item['pose_3d'] = pose_3d
        elif self.dataset == "lsp":
            # gt_parts = get_single_image_crop(self.gt_parts_path[idx], bbox, self.scale, flag=0)
            item["gt_parts"] = self.gt_parts_path[idx]
            item["gt_seg"] = self.gt_seg_path[idx]
            item["center"] = self.center[idx]
            item["orig_shape"] = self.orig_shape[idx]

        item['img'] = crop_img
        return item