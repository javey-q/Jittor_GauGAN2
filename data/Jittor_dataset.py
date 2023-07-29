"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import ipdb
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

from PIL import Image
import random
import util.util as util
import jittor as jt
import os
import json

class JittorDataset(BaseDataset):
    """ Dataset that loads images from directories
        Use option --label_dir, --image_dir to specify the directories.
        The images in the directories are sorted in alphabetical order and paired in order.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(label_nc=29)
        parser.set_defaults(contain_dontcare_label=False)

        parser.add_argument('--label_dir', type=str, required=True,
                            help='path to the directory that contains label images')
        parser.add_argument('--image_dir', type=str, default='',
                            help='path to the directory that contains photo images')
        parser.add_argument('--ref_dict', type=str, default='',
                            help='path to the file of ref_dict')
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths = self.get_paths(opt)

        util.natural_sort(label_paths)
        util.natural_sort(image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

    def get_paths(self, opt):
        print(f"label_dir: {opt.label_dir}, image_dir: {opt.image_dir}")

        self.label_dir = opt.label_dir
        label_paths = make_dataset(
            self.label_dir, recursive=False, read_cache=True)
        if len(opt.image_dir) > 0:
            self.image_dir = opt.image_dir
            image_paths = make_dataset(
                self.image_dir, recursive=False, read_cache=True)
        else:
            image_paths = []
        if opt.isTrain:
            self.isTrain = True
            assert len(label_paths) == len(
                image_paths), "The #images in %s and %s do not match. Is there something wrong?"
        else:
            self.isTrain = False

        if len(opt.ref_dict)>0:
            with open(opt.ref_dict, 'r') as f:
                self.ref_dict = json.load(f)
        else:
            self.ref_dict = {}
        self.name = "JittorDataset"

        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label = Image.open(label_path)
        params = get_params(self.opt, label.size)
        transform_label = get_transform(
            self.opt, params, method=Image.NEAREST, normalize=False)
        # label_tensor = transform_label(label) * 255.0
        label_tensor = transform_label(label)
        # label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc

        # input image (real images)
        if self.isTrain:
            image_path = self.image_paths[index]
            assert self.paths_match(label_path, image_path), \
                "The label_path %s and image_path %s don't match." % \
                (label_path, image_path)
            image = Image.open(image_path)
            image = image.convert('RGB')

            transform_image = get_transform(self.opt, params)
            image_tensor = transform_image(image)
        else:
            image_tensor = 0

        # if len(self.ref_dict) > 0:
        #     label_file = label_path.split("/")[-1]
        #     style_file = self.ref_dict[label_file].replace(".png",".jpg")
        #     style_path = "/".join([self.image_dir, style_file])
        # else:
        #     # random_ref = random.randint(0, len(self.image_paths)-1)
        #     random_ref = random.randint(-5,  5) + index
        #     random_ref = max(min(random_ref, len(self.image_paths)-1), 0)
        #     style_path = self.image_paths[random_ref]

        # style = Image.open(style_path)
        # style = style.convert('RGB')
        # transform_image = get_transform(self.opt, params)
        # style_tensor = transform_image(style)

        # print(f"label_path: {label_path}; img_path: {image_path}; style_path: {style_path};")

        input_dict = {'label': label_tensor,
                      'image': image_tensor,
                      # 'style': style_tensor,
                      'style': image_tensor,
                      'path': image_path if self.isTrain else label_path,
                      }
        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)
        # ipdb.set_trace()
        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size
