import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F

import cv2
import matplotlib.pyplot as plt
import sys
import os
import argparse

import datasets
import hopenet
import utils

import glob

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Head pose estimation using the Hopenet network.')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
            default=0, type=int)
    parser.add_argument('--data_dir', dest='data_dir', help='Directory path for data.',
          default='', type=str)
    parser.add_argument('--filename_list', dest='filename_list', help='Path to text file containing relative paths for every example.',
          default='', type=str)
    parser.add_argument('--snapshot_folder', dest='snapshot_folder', help='Name of model snapshot folder.',
          default='', type=str)
    parser.add_argument('--batch_size', dest='batch_size', help='Batch size.',
          default=1, type=int)
    parser.add_argument('--save_viz', dest='save_viz', help='Save images with pose cube.',
          default=False, type=bool)

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()

    cudnn.enabled = True
    gpu = args.gpu_id

    # ResNet101 with 3 outputs.
    # model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 23, 3], 66)
    # ResNet50
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    # ResNet18
    # model = hopenet.Hopenet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2], 66)

    print 'Loading snapshot list.'
    # Load snapshot
    snapshot_list = sorted(glob.glob(os.path.join(args.snapshot_folder, '*.pkl')))

    print 'Loading data.'

    # transformations = transforms.Compose([transforms.Scale(224),
    # transforms.RandomCrop(224), transforms.ToTensor()])

    transformations = transforms.Compose([transforms.Scale(224),
    transforms.RandomCrop(224), transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    pose_dataset = datasets.AFLW(args.data_dir, args.filename_list,
                                transformations)
    test_loader = torch.utils.data.DataLoader(dataset=pose_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=2)

    model.cuda(gpu)

    print 'Ready to test network.'

    output_file_name = args.snapshot_folder.split('/')[-1] + '_AFLW_preangles.txt'
    txt_output = open(os.join('output/batch_snapshots', output_file_name), 'w')

    for snapshot_path in snapshot_list:
        snapshot_name = snapshot_path.split('/')[-1].split('.')[0]
        print 'Loading snapshot ' + snapshot_name

        saved_state_dict = torch.load(snapshot_path)
        model.load_state_dict(saved_state_dict)

        # Test the Model
        model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
        total = 0
        n_margins = 20
        yaw_correct = np.zeros(n_margins)
        pitch_correct = np.zeros(n_margins)
        roll_correct = np.zeros(n_margins)

        idx_tensor = [idx for idx in xrange(66)]
        idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

        yaw_error = .0
        pitch_error = .0
        roll_error = .0

        l1loss = torch.nn.L1Loss(size_average=False)

        for i, (images, labels, name) in enumerate(test_loader):
            images = Variable(images).cuda(gpu)
            total += labels.size(0)
            label_yaw = labels[:,0].float()
            label_pitch = labels[:,1].float()
            label_roll = labels[:,2].float()

            yaw, pitch, roll, angles = model(images)

            # Binned predictions
            _, yaw_bpred = torch.max(yaw.data, 1)
            _, pitch_bpred = torch.max(pitch.data, 1)
            _, roll_bpred = torch.max(roll.data, 1)

            # Continuous predictions
            yaw_predicted = utils.softmax_temperature(yaw.data, 1)
            pitch_predicted = utils.softmax_temperature(pitch.data, 1)
            roll_predicted = utils.softmax_temperature(roll.data, 1)

            yaw_predicted = torch.sum(yaw_predicted * idx_tensor, 1).cpu()
            pitch_predicted = torch.sum(pitch_predicted * idx_tensor, 1).cpu()
            roll_predicted = torch.sum(roll_predicted * idx_tensor, 1).cpu()

            # Mean absolute error
            yaw_error += torch.sum(torch.abs(yaw_predicted - label_yaw) * 3)
            pitch_error += torch.sum(torch.abs(pitch_predicted - label_pitch) * 3)
            roll_error += torch.sum(torch.abs(roll_predicted - label_roll) * 3)

            if args.save_viz:
                name = name[0]
                cv2_img = cv2.imread(os.path.join(args.data_dir, name + '.jpg'))
                utils.plot_pose_cube(cv2_img, yaw_predicted[0] * 3 - 99, pitch_predicted[0] * 3 - 99, roll_predicted[0] * 3 - 99)
                cv2.imwrite(os.path.join('output/images', name + '.jpg'), cv2_img)

        print('Test error in degrees of the model on the ' + str(total) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f' % (yaw_error / total,
        pitch_error / total, roll_error / total))
        txt_output.write('Test error in degrees of model ' + snapshot_name + ' on the ' + str(total) +
        ' test images. Yaw: %.4f, Pitch: %.4f, Roll: %.4f \n' % (yaw_error / total,
        pitch_error / total, roll_error / total))

    txt_output.close()