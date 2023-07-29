from PIL import Image
import os
import json
import cv2
import matplotlib.pyplot as plt
from data.image_folder import make_dataset
import numpy as np
from skimage.exposure import match_histograms

ref_path = r'label_to_img.json'
image_dir = r'/home/gavin/Documents/Dataset/Jittor2023/train_resized/imgs'
label_dir = r'/home/gavin/Documents/Dataset/Jittor2023/val_A_labels_resized'
synthesize_dir = r'/home/gavin/Documents/Gavin/Jittor_GauGAN2/results/histlosstest/test_90/result_histloss_90'
transfer_dir = r'/home/gavin/Documents/Gavin/Jittor_GauGAN2/results/histlosstest/test_90/result_histloss_90_post_mix'

label_paths = make_dataset(
    label_dir, recursive=False, read_cache=True)
with open(ref_path, 'r') as f:
    ref_dict = json.load(f)

correl_list = []
i = 0
for label_path in label_paths:
    label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
    label_name = os.path.basename(label_path)
    # print(label_name)

    style_name = ref_dict[label_name]
    # print(style_name)
    style_path = os.path.join(image_dir, style_name)
    style_image = cv2.imread(style_path)
    # print(style_image.shape)

    synthesize_name = label_name.replace(".png", ".jpg")
    synthesize_path = os.path.join(synthesize_dir, synthesize_name)
    synthesize_image = cv2.imread(synthesize_path)
    # print(synthesize_image.shape)

    transfer_name = synthesize_name
    transfer_path = os.path.join(transfer_dir, transfer_name)
    transfer_img = match_histograms(synthesize_image, style_image, channel_axis=-1).astype(np.uint8)

    final_img = transfer_img * 0.5 + synthesize_image * 0.5
    cv2.imwrite(transfer_path, final_img)

    # correl = hist_compare2(synthesize_image, style_image)
    # if correl==correl:
    #     correl_list.append(correl)
    i += 1
print(i)