# Towards Single 2D Image-level Self-supervision for 3D Human Pose and Shape Estimation

## Prerequisties
The code needs the following libraries:

* Python 3.7
* Anaconda
* PyTorch 1.4.0

## Data Preparation
We use the dataset of Human3.6M, LSP, MPII, COCO 2014, MPI-INF-3DHP and YOUTUBE Collection for training, and Human3.6M, LSP 3DPW and UPI_S1H for testing.
We attach the google drive link of mask-rcnn results.

* Human 3.6M : http://vision.imar.ro/human3.6m/description.php
* Human 3.6M mask-rcnn results : https://drive.google.com/uc?id=12AkKZ6YSALNZgM4KZO3tcbescadln5Hz
```
├─[Human 3.6M path]
|   ├─images
|         S1_Directions_1.54138969_000001.jpg
|         S1_Directions_1.54138969_000026.jpg
|         ...
|   ├─masks
|         0_S1_Directions_1.54138969_000001.jpg
|         0_S1_Directions_1.54138969_000026.jpg
|         ...
```
* LSP : https://drive.google.com/uc?id=1R0JNZGs833f8y0MnsyU_ATkXQZ5ooW4z
```
├─[LSP path]
|   ├─images
|         im0001.jpg
|         im0002.jpg
|         ...
|   ├─ masks
|         0_im0001.jpg
|         0_im0002.jpg
|         ...
```
* MPII Images : http://human-pose.mpi-inf.mpg.de/#download
* MPII mask-rcnn results: https://drive.google.com/uc?id=1vU3_oOzOhGWN4M4kQApzx2jCrv3DIlAh
```
├─[MPII path]
|   ├─images
|         000001163.jpg
|         000003072.jpg
|         ...
|   ├─masks
|         0_000001163.jpg
|         0_000003072.jpg
|         ...
```
* COCO 2014 : https://cocodataset.org/#download
* COCO 2014 mask-rcnn results: https://drive.google.com/uc?id=190zdC5bG_WWzq4xXOhuCOy8NMfH_wfIv
```
├─[COCO path]
|   ├─train2014
|         COCO_train2014_000000000086.jpg
|         COCO_train2014_000000000529.jpg
|         ...
|   ├─masks
|         0_COCO_train2014_000000000086.jpg
|         0_COCO_train2014_000000000529.jpg
|         ...
```
* MPI_INF_3DHP : http://vcai.mpi-inf.mpg.de/3dhp-dataset/
* MPI_INF_3DHP mask-rcnn results: https://drive.google.com/uc?id=1XwK_TDb1YOrbw4YhJkC_2S3x_RdN_Dqz
```
├─[MPI_INF_3DHP path]
|   ├─S1
|   |   ├─Seq1
|   |   |   ├─imageFrames
|   |   |   |   ├─video_0
|                   frame_000001.jpg
|                   frame_000011.jpg
|                   ...
|   |   |   ├─masks
|   |   |   |   ├─video_0
|                   0_frame_000001.jpg
|                   0_frame_000011.jpg
|                   ...

```
* 3DPW : https://virtualhumans.mpi-inf.mpg.de/3DPW/evaluation.html
```
├─[3DPW path]
|   ├─imageFiles
|   |   ├─downtown_arguing_00
|             image_00000.jpg
|             image_00001.jpg
|             ...
```
* UPI_S1H : https://files.is.tuebingen.mpg.de/classner/up/#datasets
```
├─[UPI_S1H path]
|   ├─data
|   |   ├─lsp
|           im0001_part_segmentation.png
|           im0001_segmentation.png
|           ...
```
* YOUTUBE Collection : https://drive.google.com/uc?id=1PDr4QU9B6rUzBqYySQn4MKu0-Aeeik-V
```
├─[YOUTUBE Collection path]
|   ├─video1
|   |   ├─imageFiles
|           frame000350.jpg
|           frame000775.jpg
|           ...
|   |   ├─masks
|           0_frame000350.jpg
|           0_frame000775.jpg
|           ...
```

Download npz file, VIBE_data and other data.
```
source scripts/prepare_data.sh
```

```
├─SSPSE
│  ├─data
│  │  ├─dataset_extras
│           3dpw_test.npz
|           coco_2014_train.npz
|           ...
|           youtube_train.npz
```
We don't use LSP-extension.

## Extracting images from videos
```
python preprocessing.py
```

## Pretrained file
```
source scripts/pretrained.sh
```

## Training
Semi-supervised
```
python main.py --train 1 --output_dir semi_
```

Weakly-supervised

```
python main.py --train 1 --output_dir weakly_ --ignore_3d
```

Self-supervised

```
python main.py --train 1 --output_dir self_ --self_supervised
```
## Testing
H36M

```
python main.py --train 0 --checkpoint results/semi/save_pth/best.pth --test_dataset h36m-p2
```

3DPW
```
python main.py --train 0 --checkpoint results/semi/save_pth/best.pth --test_dataset 3dpw
```

LSP
```
python main.py --train 0 --checkpoint results/semi/save_pth/best.pth --test_dataset lsp
```
