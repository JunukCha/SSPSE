import glob, os, shutil

def copy_dataset():
    action_number = 0

    datset_folder_name = 'dataset'
    if not os.path.exists(datset_folder_name):
        os.mkdir(datset_folder_name)

    view_folder_list = glob.glob('/data/nucla/multiview_action/*')
    view_folder_list.sort()

    for view_folder in view_folder_list:
        view_folder_name = datset_folder_name + '/' + view_folder.split('/')[-1]
        if not os.path.exists(view_folder_name):
            os.mkdir(view_folder_name)

        action_folder_list = glob.glob(view_folder + '/*')
        action_folder_list.sort()

        for action_folder in action_folder_list:
            action_number += 1
            action_folder_name = view_folder_name + '/' + action_folder.split('/')[-1]
            if not os.path.exists(action_folder_name):
                os.mkdir(action_folder_name)

            rgb_file_list = glob.glob(action_folder + '/*_rgb.jpg')
            
            for rgb_file in rgb_file_list:
                shutil.copy(rgb_file, action_folder_name)
            print("{} is completed".format(action_number))


def separate_train_test_dataset(view):
    num_train_data = 0
    action_folder_list = glob.glob('dataset/{}/*'.format(view))

    if not action_folder_list:
        print('There is no data')
    else:
        if not os.path.exists('dataset/train_{}'.format(view)):
            os.mkdir('dataset/train_{}'.format(view))

        train_folder = 'dataset/train_{}/1'.format(view)
        if not os.path.exists(train_folder):
            os.mkdir(train_folder)

        if not os.path.exists('dataset/test_{}'.format(view)):
            os.mkdir('dataset/test_{}'.format(view))

        test_folder = 'dataset/test_{}/1'.format(view)
        if not os.path.exists(test_folder):
            os.mkdir(test_folder)

        for action_folder in action_folder_list:
            num_train_data += 1
            subject_id = action_folder.split('/')[-1][5:7]
            rgb_file_list = glob.glob(action_folder + '/*_rgb.jpg')

            for rgb_file in rgb_file_list:
                if subject_id != '10':
                    shutil.copy(rgb_file, train_folder)
                elif subject_id == '10':
                    shutil.copy(rgb_file, test_folder)
            print("{} is completed".format(num_train_data))



def separate_train_test_dataset_all():
    num_train_data = 0
    view_folder_list = glob.glob('dataset/view_*')

    if not view_folder_list:
        print('There is no data')
    else:
        for view_folder in view_folder_list:
            action_folder_list = glob.glob(view_folder+'/*')

            if not os.path.exists('dataset/train_all'):
                os.mkdir('dataset/train_all')

            train_folder = 'dataset/train_all/1'

            if not os.path.exists(train_folder):
                os.mkdir(train_folder)

            if not os.path.exists('dataset/test_all'):
                os.mkdir('dataset/test_all')

            test_folder = 'dataset/test_all/1'

            if not os.path.exists(test_folder):
                os.mkdir(test_folder)

            for action_folder in action_folder_list:
                num_train_data += 1
                subject_id = action_folder.split('/')[-1][5:7]
                rgb_file_list = glob.glob(action_folder + '/*_rgb.jpg')

                for rgb_file in rgb_file_list:
                    if subject_id != '10':
                        shutil.copy(rgb_file, train_folder)
                    elif subject_id == '10':
                        shutil.copy(rgb_file, test_folder)
                print("{} is completed".format(num_train_data))


if __name__ == '__main__':
    # view = input('which view do you want? ')
    separate_train_test_dataset_all()