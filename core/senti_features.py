import torch.utils.data as data
import os
import csv
import json
import numpy as np
import pandas as pd
import torch
import pdb
import time
import random
import utils
import config
from natsort import natsorted
import math


class SentiFeature(data.Dataset):
    def __init__(self, data_path, mode, modal, feature_fps, num_segments, sampling, seed=-1, supervision='point'):
        if seed >= 0:
            utils.set_seed(seed)

        self.mode = mode
        self.modal = modal
        self.feature_fps = feature_fps
        self.num_segments = num_segments
        self.sampling = sampling
        self.supervision = supervision
        self.data_path = data_path

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'logmfcc','img']: #use other modal features 
                self.feature_path.append(os.path.join(data_path, 'features', self.mode, _modal))
        else:
            self.feature_path = os.path.join(data_path, 'features', self.mode, self.modal)

        split_path = os.path.join(data_path, 'split_{}.txt'.format(self.mode))
        split_file = open(split_path, 'r')
        self.vid_list = []
        for line in split_file:
            self.vid_list.append(line.strip())
        split_file.close()

        self.fps_dict = json.load(open(os.path.join(data_path, 'fps_dict.json')))

        anno_path = os.path.join(data_path, 'gt.json')
        anno_file = open(anno_path, 'r')
        self.anno = json.load(anno_file)
        anno_file.close()
        
        self.class_name_to_idx = dict((v, k) for k, v in config.class_dict.items())        
        self.num_classes = len(self.class_name_to_idx.keys())
        self.point_path = os.path.join(data_path, 'point_gaussian', 'train')
        
        if self.supervision == 'point':
            self.mix_video_points_label()
            
            
        self.stored_info_all = {'new_dense_anno': [-1] * len(self.vid_list), 'sequence_score': [-1] * len(self.vid_list)}

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        data, vid_num_seg, sample_idx = self.get_data(index)
        label, point_anno, label_distribution = self.get_label(index, vid_num_seg, sample_idx)

        stored_info = {'new_dense_anno': self.stored_info_all['new_dense_anno'][index], 'sequence_score': self.stored_info_all['sequence_score'][index]}

        return index, data, label, point_anno, stored_info, self.vid_list[index], vid_num_seg, label_distribution

    def downsample_and_align(self,rgb, mfcc, T_rgb, D_rgb, T_mfcc=None, D_mfcc=None, keep_last=True, step=16):
        # 确保输入是numpy数组而不是内存视图
        if isinstance(rgb, np.memmap):
            rgb = np.array(rgb)
        if isinstance(mfcc, np.memmap):
            mfcc = np.array(mfcc)
        if T_mfcc is None or D_mfcc is None:
            T_mfcc, D_mfcc = mfcc.shape
        # 计算下采样后的块数
        n_blocks = T_rgb // step
        rgb_down = rgb[:n_blocks*step].reshape(n_blocks, step, D_rgb)[:, 0, :]
        
        # 处理最后一帧
        if keep_last and T_rgb % step != 0:
            rgb_down = np.vstack([rgb_down, rgb[-1:]])
        
        # 对齐MFCC特征
        # 确保索引不超过MFCC的长度
        indices = np.arange(0, min(n_blocks*step, T_mfcc), step)
        mfcc_down = mfcc[indices]
        
        # 处理最后一帧
        if keep_last and T_rgb % step != 0 and len(mfcc_down) < len(rgb_down):
            # 确保不会越界
            last_idx = min(T_mfcc - 1, len(mfcc) - 1)
            mfcc_down = np.vstack([mfcc_down, mfcc[last_idx:last_idx+1]])
        return rgb_down, mfcc_down
       

    def get_data(self, index):
        vid_name = self.vid_list[index]

        vid_num_seg = 0

        if self.modal == 'all':
            rgb_feature_path = os.path.join(self.feature_path[0], vid_name + '.npy')
            if not os.path.exists(rgb_feature_path):
                raise FileNotFoundError(f"RGB feature file not found: {rgb_feature_path}")
            rgb_feature = np.load(rgb_feature_path).astype(np.float32)

            # 检查并加载 MFCC 特征
            mfcc_feature_path = os.path.join(self.feature_path[1], vid_name + '.npy')
            if not os.path.exists(mfcc_feature_path):
                raise FileNotFoundError(f"MFCC feature file not found: {mfcc_feature_path}")
            mfcc_feature = np.load(mfcc_feature_path).astype(np.float32)

            # 检查并加载图像特征
            img_feature_path = os.path.join(self.feature_path[2], vid_name + '.npy')
            img_feature = np.load(img_feature_path).astype(np.float32)

            # 确保是numpy数组
            if not isinstance(rgb_feature, np.ndarray):
                rgb_feature = np.array(rgb_feature)
            if not isinstance(mfcc_feature, np.ndarray):
                mfcc_feature = np.array(mfcc_feature)
            # 重新获取正确的形状信息
            T_rgb, D_rgb = rgb_feature.shape
            T_mfcc, D_mfcc = mfcc_feature.shape
            rgb_feature, mfcc_feature = self.downsample_and_align(
                rgb_feature, mfcc_feature, T_rgb, D_rgb, T_mfcc, D_mfcc
            )
            T = min(rgb_feature.shape[0],mfcc_feature.shape[0],img_feature.shape[0])
            rgb_feature = rgb_feature[:T]
            mfcc_feature = mfcc_feature[:T]
            # img_feature = img_feature[:T]
            # mfcc_feature = mfcc_feature[:T*32].reshape(T,32,60)
            # transfo)rm
            # mfcc_feature = (mfcc_feature+50)/80
            vid_num_seg = rgb_feature.shape[0]
            # print(rgb_feature.shape[0],mfcc_feature.shape[0],img_feature.shape[0])
            if self.sampling == 'random':
                sample_idx = self.random_perturb(vid_num_seg)
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(vid_num_seg)
            else:
                raise AssertionError('Not supported sampling !')

            rgb_feature = rgb_feature[sample_idx]
            mfcc_feature = mfcc_feature[sample_idx]
            img_feature=img_feature[sample_idx]
            # 
            # print(rgb_feature.shape,mfcc_feature.shape,img_feature.shape)
            feature = [rgb_feature, mfcc_feature,img_feature]
            return [torch.from_numpy(feature[0]),torch.from_numpy(feature[1]),torch.from_numpy(feature[2])], vid_num_seg, sample_idx
        elif self.modal == 'rgb':
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)

            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(vid_num_seg)
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(vid_num_seg)
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

            return torch.from_numpy(feature), vid_num_seg, sample_idx

        elif self.modal == 'audio':
            feature = np.load(os.path.join(self.feature_path,
                                    vid_name + '.npy')).astype(np.float32)
            T = int(feature.shape[0]/32)
            feature = feature[:T*32].reshape(T,32,60)
            # transform
            feature = (feature+50)/80
            vid_num_seg = feature.shape[0]

            if self.sampling == 'random':
                sample_idx = self.random_perturb(vid_num_seg)
            elif self.sampling == 'uniform':
                sample_idx = self.uniform_sampling(vid_num_seg)
            else:
                raise AssertionError('Not supported sampling !')

            feature = feature[sample_idx]

            return torch.from_numpy(feature), vid_num_seg, sample_idx

    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]
        anno_list = self.anno['database'][vid_name]['annotations']            
        label = np.zeros([self.num_classes], dtype=np.float32)
        if self.mode =='test':
            for anno in anno_list:
                cls = anno['label']
                if cls == 'n':
                    label[0] = 1
                elif cls == 'p':
                    label[1] = 1
        label_distribution = np.zeros([self.num_classes], dtype=np.float32)
        
        classwise_anno = [[]] * self.num_classes

        if self.supervision == 'video':
            return label, torch.Tensor(0)

        elif self.supervision == 'point':
            temp_anno = np.zeros([vid_num_seg, self.num_classes], dtype=np.float32)
            # t_factor = self.feature_fps / (self.fps_dict[vid_name] * 16)
            t_factor = 1/16
            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][['point', 'start_frame','stop_frame','class_index']]

                # point = temp_df['point'][key]
                # start=temp_df['start_frame'][key]
                # end=temp_df['stop_frame'][key]
                # class_idx = temp_df['class_index'][key]
                # # label smoot
                # temp_anno[int(point * t_factor)] = 0.1
                # temp_anno[int(point * t_factor)][class_idx] = 0.9
                # temp_anno[int(start * t_factor):int(end * t_factor)][class_idx]=0.9
                # # MLL
                # label[class_idx] = 1
                # # smoothed MLL
                # label_distribution[class_idx] += 1
            temp_gt = np.zeros([vid_num_seg, self.num_classes], dtype=np.float32)
            # t_factor = self.feature_fps / (self.fps_dict[vid_name] * 16)
            t_factor = 1/16
            temp_df = self.point_anno[self.point_anno["video_id"] == vid_name][[ 'class_index','start_frame','stop_frame']]
            '''
            point_label:T*C ,[0,1]有标签的位置会去做一个[1,0],其他位置[0,0]
            '''
            for key in temp_df['start_frame'].keys():
                start=temp_df['start_frame'][key]
                end=temp_df['stop_frame'][key]
                class_idx = temp_df['class_index'][key]
                # label smooth
                indexstart=int(start * t_factor)
                indexend=int(end * t_factor)
                if indexstart < 0 or indexend >= vid_num_seg:
                    indexend=vid_num_seg
                temp_gt[indexstart:indexend,:]=0.1
                temp_gt[indexstart:indexend+1, class_idx] = 0.9
                label[class_idx] = 1
                # smoothed MLL
                label_distribution[class_idx] += 1   
            point_label = temp_gt[sample_idx, :]
            label_distribution = label_distribution/label_distribution.sum()
            return label, torch.from_numpy(point_label), label_distribution

    def mix_video_points_label(self,n=0):
        # annotxts = []
        # for annot in os.listdir(self.point_path):
        #     annotxts.append(np.loadtxt(os.path.join(self.point_path,annot),dtype=str))

        # vid_names = []
        # vid_anno_ids=[]
        # for annotxt in annotxts:
        #     vid_anno_dict={}
        #     for anno in annotxt[1:]:
        #         vid_name = anno.split(',')[4]
        #         if vid_name not in vid_names:
        #             vid_names.append(vid_name)
                
        #         if vid_name not in vid_anno_dict.keys():
        #             vid_anno_dict[vid_name]=[anno]
        #         else:
        #             vid_anno_dict[vid_name].append(anno)
        #     vid_anno_ids.append(vid_anno_dict)
            
        # with open(os.path.join(self.point_path,'../point_labels.csv'),'w+')as txt:
        #     txt.write('class,class_index,start_frame,stop_frame,video_id,point\n')
        #     for vid_name in natsorted(vid_names):
        #         cur_id = (random.randint(0,len(vid_anno_ids)-1)+random.randint(0,n))%len(vid_anno_ids)
        #         vid_anno_dict = vid_anno_ids[cur_id]
        #         vid_anno = vid_anno_dict[vid_name]
                
        #         for anno in vid_anno:
        #             txt.write(anno+'\n')
        # os.path.join(data_path, 'point_gaussian', 'train')
        self.point_anno = pd.read_csv(os.path.join(self.data_path,'point_label.csv'))
            
    
    def random_perturb(self, length):
        if self.num_segments == length or self.num_segments == -1:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        for i in range(self.num_segments):
            if i < self.num_segments - 1:
                if int(samples[i]) != int(samples[i + 1]):
                    samples[i] = np.random.choice(range(int(samples[i]), int(samples[i + 1]) + 1))
                else:
                    samples[i] = int(samples[i])
            else:
                if int(samples[i]) < length - 1:
                    samples[i] = np.random.choice(range(int(samples[i]), length))
                else:
                    samples[i] = int(samples[i])
        return samples.astype(int)


    def uniform_sampling(self, length):
        if length <= self.num_segments or self.num_segments == -1:
            return np.arange(length).astype(int)
        samples = np.arange(self.num_segments) * length / self.num_segments
        samples = np.floor(samples)
        return samples.astype(int)