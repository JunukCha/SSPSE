import os
import sys
import cv2

    
def train_data(dataset_path):

    # training data
    user_list = range(1,9)
    seq_list = range(1,3)
    vid_list = list(range(3)) + list(range(4,9))

    for user_i in user_list:
        for seq_i in seq_list:
            seq_path = os.path.join(dataset_path,
                                    'S' + str(user_i),
                                    'Seq' + str(seq_i))
            
            for j, vid_i in enumerate(vid_list):
                # image folder
                imgs_path = os.path.join(seq_path,    
                                         'imageFrames',
                                         'video_' + str(vid_i))

                # extract frames from video file
                # if doesn't exist
                if not os.path.isdir(imgs_path):
                    os.makedirs(imgs_path)

                # video file
                vid_file = os.path.join(seq_path,
                                        'imageSequence',
                                        'video_' + str(vid_i) + '.avi')
                vidcap = cv2.VideoCapture(vid_file)

                # process video
                frame = 0
                while 1:
                    # extract all frames
                    success, image = vidcap.read()
                    if not success:
                        break
                    frame += 1
                    # image name
                    imgname = os.path.join(imgs_path,
                        'frame_%06d.jpg' % frame)
                    # save image
                    cv2.imwrite(imgname, image)