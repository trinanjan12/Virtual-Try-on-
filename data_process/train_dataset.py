# from data_process.train_dataset import RegularDataset
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import os.path as osp
from PIL import Image
import numpy as np
from torchvision import transforms
from torchvision import utils
from utils import pose_utils
from PIL import ImageDraw
# from utils.transforms import create_part
import time
import json
import random
import cv2
from torch.utils.tensorboard import SummaryWriter
np.seterr(divide='ignore', invalid='ignore')


class RegularDatasetDensepose(Dataset):
    def __init__(self, file_name, file_type, data_folder, augment):
        self.transforms = augment
        pair_list = [i.strip() for i in open(file_name, 'r').readlines()]
        train_list = list(
            filter(lambda p: p.split('\t')[3] == file_type, pair_list))
        train_list = [i for i in train_list]
        self.size = (256, 192)
        self.img_list = train_list
        self.data_folder = data_folder

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        t0 = time.time()
        img_ext = '.jpg'
        try:
            img_source = self.img_list[index].split('\t')[0]
            img_target = self.img_list[index].split('\t')[1]
            cloth_img = self.img_list[index].split('\t')[2]
        except:
            img_source = self.img_list[index].split(' ')[0]
            img_target = self.img_list[index].split(' ')[1]
            cloth_img = self.img_list[index].split(' ')[2]

        source_splitext = os.path.splitext(img_source)[0]
        target_splitext = os.path.splitext(img_target)[0]
        cloth_splitext = os.path.splitext(cloth_img)[0]

        # png or jpg
        source_img_path = os.path.join('datasets/zalando/source_images',
                                       source_splitext + img_ext)
        target_img_path = os.path.join('datasets/zalando/target_images',
                                       target_splitext + img_ext)
        cloth_img_path = os.path.join('datasets/zalando/cloth', cloth_img)
        cloth_parse_path = os.path.join('datasets/zalando/cloth_mask',
                                        cloth_splitext + img_ext)
        warped_cloth_name = cloth_img

        # image
        warped_cloth_path = os.path.join('dataset', 'cloth', warped_cloth_name)
        source_img = self.open_transform(source_img_path, False)
        target_img = self.open_transform(target_img_path, False)
        cloth_img = self.open_transform(cloth_img_path, False)
        cloth_parse = self.parse_cloth(cloth_parse_path)

        try:
            warped_cloth_parse_name = cloth_img
            # mask
            warped_cloth_parse_path = os.path.join('dataset', 'cloth_mask',
                                                   warped_cloth_parse_name)
            warped_cloth_parse = self.parse_cloth(warped_cloth_parse_path)
        except:
            warped_cloth_parse = torch.ones(1, 256, 192)

        if os.path.exists(warped_cloth_path):
            warped_cloth_img = self.open_transform(warped_cloth_path, False)
        else:
            warped_cloth_img = cloth_img

        # parsing
        source_parse_vis_path = os.path.join('datasets/zalando/parse_cihp_source',
                                             source_splitext + '_vis' + '.png')
        target_parse_vis_path = os.path.join('datasets/zalando/parse_cihp_target',
                                             target_splitext + '_vis' + '.png')
        source_parse_vis = self.transforms['2'](
            Image.open(source_parse_vis_path))
        target_parse_vis = self.transforms['2'](
            Image.open(target_parse_vis_path))

        source_parse_path = os.path.join('datasets/zalando/parse_cihp_source',
                                         source_splitext + '.png')
        target_parse_path = os.path.join('datasets/zalando/parse_cihp_target',
                                         target_splitext + '.png')

        source_parse = pose_utils.parsing_embedding(source_parse_path)
        source_parse_tformed = self.custom_transform(source_parse, True)
        source_parse = torch.from_numpy(source_parse)

        target_parse = pose_utils.parsing_embedding(target_parse_path)
        target_parse_tformed = self.custom_transform(target_parse, True)
        target_parse = torch.from_numpy(target_parse)

        source_parse_shape = np.array(Image.open(source_parse_path))
        source_parse_shape = (source_parse_shape > 0).astype(np.float32)
        source_parse_shape = Image.fromarray(
            (source_parse_shape * 255).astype(np.uint8))
        source_parse_shape = source_parse_shape.resize(
            (self.size[1] // 16, self.size[0] // 16),
            Image.BILINEAR)  # downsample and then upsample
        source_parse_shape = source_parse_shape.resize(
            (self.size[1], self.size[0]), Image.BILINEAR)
        source_parse_shape = self.transforms['2'](source_parse_shape)  # [-1,1]

        source_parse_head = (np.array(Image.open(source_parse_path)) == 1).astype(np.float32) + \
            (np.array(Image.open(source_parse_path)) == 2).astype(np.float32) + \
            (np.array(Image.open(source_parse_path)) == 4).astype(np.float32) + \
            (np.array(Image.open(source_parse_path)) == 13).astype(np.float32)

        target_parse_cloth = (np.array(Image.open(target_parse_path)) == 5).astype(np.float32) + \
            (np.array(Image.open(target_parse_path)) == 6).astype(np.float32) + \
            (np.array(Image.open(target_parse_path)) == 7).astype(np.float32)
        target_parse_cloth = torch.from_numpy(target_parse_cloth)
        # prepare for warped cloth

        phead = torch.from_numpy(source_parse_head)  # [0,1]
        pcm = target_parse_cloth  # [0,1]
        im = target_img  # [-1,1]
        im_c = im * pcm + (
            1 - pcm)  # [-1,1], fill 1 for other parts --> white same as GT ...
        im_h = source_img * phead - (
            1 - phead
        )  # [-1,1], fill -1 for other parts, thus become black visual

        # pose heatmap embedding

        # source_pose_path = os.path.join('datasets/zalando/source_keypoints',
        #                                 source_splitext + '_keypoints.npy')
        # source_pose_org = np.load(source_pose_path)
        # source_pose = []
        # for i in source_pose_org:
        #     source_pose.extend(i)
        # source_pose_loc = pose_utils.pose2loc(source_pose)
        # source_pose_embedding = pose_utils.heatmap_embedding(
        #     self.size, source_pose_loc)

        # target_pose_path = os.path.join('datasets/zalando/target_keypoints',
        #                                 target_splitext + '_keypoints.npy')
        # target_pose_org = np.load(target_pose_path)
        # target_pose = []
        # for i in target_pose_org:
        #     target_pose.extend(i)
        # target_pose_loc = pose_utils.pose2loc(target_pose)
        # target_pose_embedding = pose_utils.heatmap_embedding(
        #     self.size, target_pose_loc)
        # target_pose_img, _ = pose_utils.draw_pose_from_cords(
        #     target_pose_loc, (256, 192))

        # pose heatmap embedding
        source_pose_path = os.path.join(
            'datasets/zalando/source_keypoints', source_splitext + '_keypoints.json')
        with open(source_pose_path, 'r') as f:
            a = json.load(f)
            source_pose = a['people'][0]['pose_keypoints_2d']
        source_pose_loc = pose_utils.pose2loc(source_pose)
        source_pose_embedding = torch.from_numpy(
            pose_utils.heatmap_embedding(self.size, source_pose_loc))

        target_pose_path = os.path.join(
            'datasets/zalando/target_keypoints', target_splitext + '_keypoints.json')
        with open(target_pose_path, 'r') as f:
            a = json.load(f)
            target_pose = a['people'][0]['pose_keypoints_2d']
        target_pose_loc = pose_utils.pose2loc(target_pose)
        target_pose_embedding = torch.from_numpy(
            pose_utils.heatmap_embedding(self.size, target_pose_loc))
        # target_pose_img, _ = pose_utils.draw_pose_from_cords(
        #     target_pose_loc, (256, 192))

        # Densepose preprocess source
        source_densepose_path = os.path.join('datasets/zalando/densepose_numpy_source/',
                                             source_splitext + '.npy')
        source_densepose_data = np.load(
            source_densepose_path).astype('uint8')  # (256,192,3)
        source_densepose_parts_embeddings = self.parsing_embedding(
            source_densepose_data[:, :, 0])
        source_densepose_parts_embeddings = np.transpose(
            source_densepose_parts_embeddings, axes=(1, 2, 0))
        source_densepose_data_final = np.concatenate(
            (source_densepose_parts_embeddings, source_densepose_data[:, :, 1:]), axis=-1)  # channel(27), H, W
        source_densepose_data_final = torch.from_numpy(
            np.transpose(source_densepose_data_final, axes=(2, 0, 1)))

        # Densepose preprocess target
        target_densepose_path = os.path.join('datasets/zalando/densepose_numpy_target/',
                                             source_splitext + '.npy')
        target_densepose_data = np.load(
            target_densepose_path).astype('uint8')  # (256,192,3)
        target_densepose_parts_embeddings = self.parsing_embedding(
            target_densepose_data[:, :, 0])
        target_densepose_parts_embeddings = np.transpose(
            target_densepose_parts_embeddings, axes=(1, 2, 0))
        target_densepose_data_final = np.concatenate(
            (target_densepose_parts_embeddings, target_densepose_data[:, :, 1:]), axis=-1)  # channel(27), H, W
        target_densepose_data_final = torch.from_numpy(
            np.transpose(target_densepose_data_final, axes=(2, 0, 1)))

        result = {
            'source_parse': source_parse,
            'source_parse_tformed' : source_parse_tformed,
            'target_parse': target_parse,
            'target_parse_tformed' : target_parse_tformed,
            'source_parse_vis': source_parse_vis,
            'target_parse_vis': target_parse_vis,
            'source_pose_embedding': source_pose_embedding,
            'target_pose_embedding': target_pose_embedding,
            'target_pose_loc': target_pose_loc,
            'source_image': source_img,
            'target_image': target_img,
            'cloth_image': cloth_img,
            'cloth_parse': cloth_parse,
            'source_parse_shape': source_parse_shape,
            'im_h': im_h,  # source image head and hair
            'im_c': im_c,  # target_cloth_image_warped
            'source_image_name': source_splitext + img_ext,
            'target_image_name': target_splitext + img_ext,
            'cloth_image_name': cloth_splitext + img_ext,
            'warped_cloth_image': warped_cloth_img,
            'warped_cloth_name': warped_cloth_name,
            'warped_cloth_path': warped_cloth_path,
            'source_img_path': source_img_path,
            'target_img_path': target_img_path,
            'target_pose_path': target_pose_path,
            'target_parse_path': target_parse_path,
            'source_parse_vis_path': source_parse_vis_path,
            'target_parse_vis_path': target_parse_vis_path,
            # 'target_pose_img': target_pose_img,
            'warped_cloth_parse': warped_cloth_parse,
            'target_parse_cloth': target_parse_cloth,
            'source_densepose_path': source_densepose_path,
            'source_densepose_data': source_densepose_data_final,
            'target_densepose_path': target_densepose_path,
            'target_densepose_data': target_densepose_data_final
        }

        return result

    def open_transform(self, path, downsample=False):
        img = Image.open(path)
        if downsample:
            img = img.resize((96, 128), Image.BICUBIC)
        img = self.transforms['2'](img)
        return img

    def parse_cloth(self, path, downsample=False):
        cloth_parse = Image.open(path)
        cloth_parse_array = np.array(cloth_parse)
        cloth_parse = (cloth_parse_array == 255).astype(np.float32)  # 0 | 1
        cloth_parse = cloth_parse[np.newaxis, :]

        if downsample:
            [X, Y] = np.meshgrid(range(0, 192, 2), range(0, 256, 2))
            cloth_parse = cloth_parse[:, Y, X]

        cloth_parse = torch.from_numpy(cloth_parse)

        return cloth_parse

    def parsing_embedding(self, parse_obj):
        parse = np.array(parse_obj)
        parse_channel = 25
        parse_emb = []
        for i in range(parse_channel):
            parse_emb.append((parse == i).astype(np.float32).tolist())
        parse = np.array(parse_emb).astype(np.float32)
        return parse

    def custom_transform(self, input_image, per_channel_transform):

        if per_channel_transform:
            num_channel_image = input_image.shape[0]
            tform_input_image_np = np.zeros(
                shape=input_image.shape, dtype=input_image.dtype)

            for i in range(num_channel_image):
                # TODO check why i!=5 makes a big difference in the output
                if i != 1 and i != 2 and i != 4 and i != 5 and i != 13:
                    # if i != 0 and i != 1 and i != 2 and i != 4 and i != 13:
                    tform_input_image_np[i] = self.transforms['1'](
                        input_image[i])
                else:
                    tform_input_image_np[i] = self.transforms['3'](
                        input_image[i])

        return torch.from_numpy(tform_input_image_np)


if __name__ == '__main__':
    pass
