import os
import sys
import cv2
import glob
import cdflib

# Illustrative script for training data extraction
# No SMPL parameters will be included in the .npz file.
def h36m_train_extract(dataset_path):

    # users in validation set
    user_list = [1, 5, 6, 7, 8]

    # go over each user
    for user_i in user_list:
        user_name = 'S%d' % user_i
        # path with GT 3D pose
        pose_path = os.path.join(dataset_path, user_name, 'MyPoseFeatures', 'D3_Positions_mono')
        # path with videos
        vid_path = os.path.join(dataset_path, user_name, 'Videos')

        # go over all the sequences of each user
        seq_list = glob.glob(os.path.join(pose_path, '*.cdf'))
        seq_list.sort()
        for seq_i in seq_list:
            # sequence info
            seq_name = seq_i.split('/')[-1]
            action, camera, _ = seq_name.split('.')
            action = action.replace(' ', '_')
            # irrelevant sequences
            if action == '_ALL':
                continue

            # 3D pose file
            poses_3d = cdflib.CDF(seq_i)['Pose'][0]

            # video file
            vid_file = os.path.join(vid_path, seq_name.replace('cdf', 'mp4'))
            imgs_path = os.path.join(dataset_path, 'images')
            vidcap = cv2.VideoCapture(vid_file)

            # go over each frame of the sequence
            for frame_i in range(poses_3d.shape[0]):
                # read video frame
                success, image = vidcap.read()
                if not success:
                    break

                # check if you can keep this frame
                if frame_i % 25 == 0:
                    # image name
                    imgname = '%s_%s.%s_%06d.jpg' % (user_name, action, camera, frame_i+1)
                    
                    # save image
                    img_out = os.path.join(imgs_path, imgname)
                    cv2.imwrite(img_out, image)