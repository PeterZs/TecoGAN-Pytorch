from utils.img_utils import Distortion_v2
from easydict import EasyDict as edict
import yaml
import cv2
import random
import os
import shutil


yaml_path = r'./train_desktop.yaml'

f = open(yaml_path, 'r')
config = edict(yaml.load(f))

input_dir = r'/home/SENSETIME/qianjinhao/program_project/data/VimeoTecoGAN/GT'

output_d_dir = r'/home/SENSETIME/qianjinhao/program_project/data/FaceTest/Degrade'
if not os.path.exists(output_d_dir):
    os.mkdir(output_d_dir)

output_gt_dir = r'/home/SENSETIME/qianjinhao/program_project/data/FaceTest/GT'
if not os.path.exists(output_gt_dir):
    os.mkdir(output_gt_dir)

dirs = os.listdir(input_dir)

for dir in dirs:
    if int(dir[-4:]) < 1300:
        continue
    dir = os.path.join(input_dir,dir)
    gt_images = os.listdir(dir)
    seed_ = random.randint(1, 1000)

    gt_dir = os.path.join(output_gt_dir, dir.split('/')[-1])
    # Copy directory tree (cp -R src dst)
    shutil.copytree(dir, gt_dir)

    d_dir = os.path.join(output_d_dir, dir.split('/')[-1])
    if not os.path.exists(d_dir):
        os.mkdir(d_dir)


    for gt_image_path in gt_images:
        gt_image = cv2.imread(os.path.join(dir,gt_image_path))
        Size = (gt_image.shape[1], gt_image.shape[0]) # (W, H)
        Distort = Distortion_v2(config.pair_data.distortion, 1)
        gt_raw, d_image, gain \
            = Distort.Distort_random_v5(gt_image, Size,seed_)

        cv2.imwrite(os.path.join(d_dir,gt_image_path),d_image)
