import sys, os, argparse
import cv2
import mediapipe as mp
import numpy as np
import time
import math
from scipy.spatial import distance
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
import torchvision
import torch.nn.functional as F
from PIL import Image
import datasets, hopenet, utils
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
from skimage import io
import dlib

if __name__ == '__main__':
    cudnn.enabled = True

    batch_size = 1
    gpu = 0
    snapshot_path = "C:\\Users\\Flamingo\\Desktop\\毕设\\code\\deep-head-pose\\output\\snapshots\\hopenet_robust_alpha1.pkl"
    out_dir = 'output/video'
    video_path = ""

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # ResNet50 structure
    model = hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3],
                            66)
    print('Loading snapshot.')
    # Load snapshot
    saved_state_dict = torch.load(snapshot_path)
    model.load_state_dict(saved_state_dict)

    print('Loading data.')

    transformations = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    model.cuda(gpu)
    print('Ready to test network.')

    # Test the Model
    model.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    total = 0

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda(gpu)

    # mediapipe face detection model
    face_detection = mp.solutions.face_detection
    face_mesh = mp.solutions.face_mesh
    cap = cv2.VideoCapture(0)
    # cap = cv2.VideoCapture('./video/test.mp4')
    with face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.6) as face_detection:
        while cap.isOpened():
            start = time.time()
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                # continue
                continue
            image = cv2.flip(image, 1)
            img_h, img_w, img_c = image.shape
            # To improve performance, optionally mark the image as not writeable to
            image.flags.writeable = False
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = face_detection.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.detections:  # face detected
                # get bounding box
                detection = results.detections
                bbox = detection[0].location_data.relative_bounding_box
                bbox_points = {
                    "xmin": int(bbox.xmin * img_w),
                    "ymin": int(bbox.ymin * img_h),
                    "xmax": int(bbox.width * img_w + bbox.xmin * img_w),
                    "ymax": int(bbox.height * img_h + bbox.ymin * img_h),
                    "box_width": int(bbox.width),
                    "box_height": int(bbox.height)
                }
                xmin = int(bbox.xmin * img_w)
                ymin = int(bbox.ymin * img_h)
                xmax = int(bbox.width * img_w + bbox.xmin * img_w)
                ymax = int(bbox.height * img_h + bbox.ymin * img_h)
                bbox_width = int(bbox.width)
                bbox_height = int(bbox.height)

                bbox_width = abs(xmax - xmin)
                bbox_height = abs(ymax - ymin)
                xmin -= 2 * bbox_width / 4
                xmax += 2 * bbox_width / 4
                ymin -= 3 * bbox_height / 4
                ymax += bbox_height / 4
                xmin = int(max(xmin, 0))
                ymin = int(max(ymin, 0))
                xmax = int(min(img_h, xmax))
                ymax = int(min(img_w, ymax))
                # Crop image
                img_RGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)[ymin:ymax,
                                                                 xmin:xmax]
                img_RGB = Image.fromarray(img_RGB)

                # Transform
                img_RGB = transformations(img_RGB)
                img_shape = img_RGB.size()
                img_RGB = img_RGB.view(1, img_shape[0], img_shape[1],
                                       img_shape[2])
                img_RGB = Variable(img_RGB).cuda(gpu)

                yaw, pitch, roll = model(img_RGB)

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(
                    yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(
                    pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(
                    roll_predicted.data[0] * idx_tensor) * 3 - 99

                # utils.plot_pose_cube(frame, yaw_predicted, pitch_predicted, roll_predicted, (x_min + x_max) / 2, (y_min + y_max) / 2, size = bbox_width)
                utils.draw_axis(image,
                                yaw_predicted,
                                pitch_predicted,
                                roll_predicted,
                                tdx=(xmin + xmax) / 2,
                                tdy=(ymin + ymax) / 2,
                                size=bbox_height / 2)
                # Plot expanded bounding box
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0),
                              1)
                end = time.time()
                total = end - start
                fps = 1 / total
                cv2.putText(
                    image,
                    f"roll:{int(roll_predicted)} pitch:{int(pitch_predicted)} yaw:{int(yaw_predicted)}",
                    (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                cv2.putText(image, f'FPS: {int(fps)}', (0, 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.imshow('MediaPipe Face Mesh', image)

            if cv2.waitKey(1) & 0xFF == 27:
                break
    cap.release()
