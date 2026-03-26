import torch
import torch.nn as nn
import numpy as np
from scipy.interpolate import interp1d
import os
import sys
import random
import config


def upgrade_resolution(arr, scale):
    x = np.arange(0, arr.shape[0]) # 0 ~ T
    f = interp1d(x, arr, kind='linear', axis=0, fill_value='extrapolate')
    scale_x = np.arange(0, arr.shape[0], 1 / scale)
    up_scale = f(scale_x)
    return up_scale

def norm(data):
    l2 = torch.norm(data, p = 2, dim = -1, keepdim = True)
    return torch.div(data, l2)

def get_proposal_oic(tList, wtcam, final_score, c_pred, scale, v_len, sampling_frames, num_segments, _lambda=0.03, gamma=0.3):
    t_factor = float(16 * v_len) / (scale * num_segments * sampling_frames)
    # ourfactor
    # t_factor = float(v_len) / (scale * num_segments * sampling_frames)
    temp = []
    for i in range(len(tList)):
        # i: class_ind  tList: Cx1xTseg
        c_temp = []
        temp_list = np.array(tList[i])[0] # Tseg
        if temp_list.any():
            grouped_temp_list = grouping(temp_list)
            for j in range(len(grouped_temp_list)):
                inner_score = np.mean(wtcam[grouped_temp_list[j], i, 0])

                len_proposal = len(grouped_temp_list[j])
                outer_s = max(0, int(grouped_temp_list[j][0] - _lambda * len_proposal))
                outer_e = min(int(wtcam.shape[0] - 1), int(grouped_temp_list[j][-1] + _lambda * len_proposal))

                outer_temp_list = list(range(outer_s, int(grouped_temp_list[j][0]))) + list(range(int(grouped_temp_list[j][-1] + 1), outer_e + 1))
                
                if len(outer_temp_list) == 0:
                    outer_score = 0
                else:
                    outer_score = np.mean(wtcam[outer_temp_list, i, 0])

                c_score = inner_score - outer_score + gamma * final_score[c_pred[i]]
                t_start = grouped_temp_list[j][0] * t_factor
                t_end = (grouped_temp_list[j][-1] + 1) * t_factor
                c_temp.append([c_pred[i], c_score, t_start, t_end])
            temp.append(c_temp)
    return temp


def result2json(result):
    result_file = []    
    for i in range(len(result)):
        line = {'label': config.class_dict[result[i][0]], 'score': result[i][1],
                'segment': [result[i][2], result[i][3]]}
        result_file.append(line)
    return result_file


def grouping(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)


def save_best_record(test_info, file_path):
    fo = open(file_path, "a+")
    fo.write("Step: {}\n".format(test_info["step"][-1]))
    fo.write("average_mAP[0.1:0.3]: {:.4f}\n".format(test_info["average_mAP[0.1:0.3]"][-1]))
    fo.write("\n")
    fo.write("average_pAP[0.1:0.3]: {:.4f}\n".format(test_info["average_pAP[0.1:0.3]"][-1]))
    fo.write("average_nAP[0.1:0.3]: {:.4f}\n".format(test_info["average_nAP[0.1:0.3]"][-1]))
    fo.write("\n")

    tIoU_thresh = np.linspace(0.1, 0.3, 5)
    for i in range(len(tIoU_thresh)):
        fo.write("mAP@{:.2f}: {:.4f}\n".format(tIoU_thresh[i], test_info["mAP@{:.2f}".format(tIoU_thresh[i])][-1]))
    fo.write("\n")
    RcAVG=0
    for i in range(len(tIoU_thresh)):
        RcAVG+=test_info["Rc@{:.2f}".format(tIoU_thresh[i])][-1]
        fo.write("Rc@{:.2f}: {:.4f}\n".format(tIoU_thresh[i], test_info["Rc@{:.2f}".format(tIoU_thresh[i])][-1]))
    fo.write("\n")
    F2AVG=0
    for i in range(len(tIoU_thresh)):
        F2AVG+=test_info["F2@{:.2f}".format(tIoU_thresh[i])][-1]
        fo.write("F2@{:.2f}: {:.4f}\n".format(tIoU_thresh[i], test_info["F2@{:.2f}".format(tIoU_thresh[i])][-1]))
    fo.write("\n")
    fo.write("Rc@AVG: {:.4f}\n".format(RcAVG/5))
    fo.write("F2@AVG: {:.4f}\n".format(F2AVG/5))
    fo.write("\n")
    fo.write("\n")    
    
    fo.close()


def minmax_norm(act_map, min_val=None, max_val=None):
    if min_val is None or max_val is None:
        relu = nn.ReLU()
        max_val = relu(torch.max(act_map, dim=1)[0])
        min_val = relu(torch.min(act_map, dim=1)[0])

    delta = max_val - min_val
    delta[delta <= 0] = 1
    ret = (act_map - min_val) / delta.detach()

    ret[ret > 1] = 1
    ret[ret < 0] = 0

    return ret


def nms(proposals, thresh):
    proposals = np.array(proposals)
    x1 = proposals[:, 2]
    x2 = proposals[:, 3]
    scores = proposals[:, 1]

    areas = x2 - x1 + 1
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]

        keep.append(proposals[i].tolist())
        xx1 = np.maximum(x1[i], x1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])

        inter = np.maximum(0.0, xx2 - xx1 + 1)

        iou = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(iou < thresh)[0]
        order = order[inds + 1]
        
    return keep


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic=True
    torch.backends.cudnn.benchmark=False

def extract_region_feat(seq, embeded_feature): 
    '''
    Extract region features.
    Input: seq:[0,1,1,0,...,0,1,1,0] embeded_feature: [T,F]
    Output: feature list:[[T1,F],[T2,F],...]
    '''
    seq_diff = seq[1:] - seq[:-1]
    range_idx = torch.nonzero(seq_diff).squeeze(1)
    range_idx = range_idx.cpu().data.numpy().tolist()
    if len(range_idx) == 0:
        return
    if seq_diff[range_idx[0]] != 1:
        range_idx = [-1] + range_idx
    if seq_diff[range_idx[-1]] != -1:
        range_idx = range_idx + [seq_diff.shape[0] - 1]

    feature_lsts = []
    idx = []
    for i in range(len(range_idx) // 2):
        if range_idx[2 * i + 1] - range_idx[2 * i] < 1:
            continue
        feature_lsts.append(embeded_feature[range_idx[2 * i] + 1:range_idx[2 * i + 1] + 1].clone())
        idx.append([range_idx[2 * i] + 1, range_idx[2 * i + 1] + 1])
    return feature_lsts
def save_config(config, file_path):
    fo = open(file_path, "w")
    fo.write("Configurtaions:\n")
    fo.write(str(config))
    fo.close()


def feature_sampling(features, start, end, num_divide):
    step = (end - start) / num_divide

    feature_lst = torch.zeros((num_divide, features.shape[1])).cuda()
    for i in range(num_divide):
        start_point = int(start + step * i)
        end_point = int(start + step * (i+1))
        
        if start_point >= end_point:
            end_point += 1

        sample_id = np.random.randint(start_point, end_point)

        feature_lst[i] = features[sample_id]

    return feature_lst.mean(dim=0)


def get_oic_score(cas_sigmoid_fuse, start, end, delta=0.25):
    length = end - start + 1

    inner_score = torch.mean(cas_sigmoid_fuse[start:end+1])
    
    outer_s = max(0, int(start - delta * length))
    outer_e = min(int(cas_sigmoid_fuse.shape[0] - 1), int(end + delta * length))

    outer_seg = list(range(outer_s, start)) + list(range(end + 1, outer_e + 1))

    if len(outer_seg) == 0:
        outer_score = 0
    else:
        outer_score = torch.mean(cas_sigmoid_fuse[outer_seg])

    return inner_score - outer_score


def select_seed_act_score(cas_sigmoid_fuse, point_anno):
    point_anno_agnostic = point_anno.max(dim=2)[0]
    bkg_seed = torch.zeros_like(point_anno_agnostic)
    act_seed = point_anno.clone().detach()

    bkg_thresh = 0.95
    bkg_score = cas_sigmoid_fuse[:,:,-1]

    for b in range(point_anno.shape[0]):
        act_idx = torch.nonzero(point_anno_agnostic[b]).squeeze(1) # index of point
        """ most left """
        if act_idx[0] > 0:
            bkg_score_tmp = bkg_score[b,:act_idx[0]]
            idx_tmp = bkg_seed[b,:act_idx[0]]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = bkg_score_tmp.max().detach()

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[:start_index] = bkg_score_tmp.max().detach()
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[:max_index+1] = bkg_score_tmp.max().detach()

            """ pseudo action point selection """
            for j in range(act_idx[0] - 1, -1, -1):
                if bkg_score[b][j] <= torch.max(cas_sigmoid_fuse[b][j][:-1]) and bkg_seed[b][j] < 1:
                    act_seed[b, j] = cas_sigmoid_fuse[b, j]
                    # act_anno[b, j] = act_anno[b, act_idx[0]]
                else:
                    break

        """ most right """
        if act_idx[-1] < (point_anno.shape[1] - 1):
            bkg_score_tmp = bkg_score[b,act_idx[-1]+1:]
            idx_tmp = bkg_seed[b,act_idx[-1]+1:]
            idx_tmp[bkg_score_tmp >= bkg_thresh] = bkg_score_tmp.max().detach()

            if idx_tmp.sum() >= 1:
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                idx_tmp[start_index:] = bkg_score_tmp.max().detach()
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index:] = bkg_score_tmp.max().detach()

            """ pseudo action point selection """
            for j in range(act_idx[-1] + 1, point_anno.shape[1]):
                if bkg_score[b][j] <= torch.max(cas_sigmoid_fuse[b][j][:-1]) and bkg_seed[b][j] < 1:
                    act_seed[b, j] = cas_sigmoid_fuse[b, j]
                    # act_anno[b, j] = act_anno[b, act_idx[-1]]
                else:
                    break
            
        """ between two instances """
        for i in range(len(act_idx) - 1):
            if act_idx[i+1] - act_idx[i] <= 1:
                continue

            bkg_score_tmp = bkg_score[b,act_idx[i]+1:act_idx[i+1]] #B T 1
            idx_tmp = bkg_seed[b,act_idx[i]+1:act_idx[i+1]] #B T C+1
            idx_tmp[bkg_score_tmp >= bkg_thresh] = bkg_score_tmp.max().detach()

            if idx_tmp.sum() >= 2: # 多个背景点
                start_index = idx_tmp.nonzero().squeeze(1)[0]
                end_index = idx_tmp.nonzero().squeeze(1)[-1]
                idx_tmp[start_index+1:end_index] = bkg_score_tmp.max().detach()                                   
            else:
                max_index = bkg_score_tmp.argmax(dim=0)
                idx_tmp[max_index] = bkg_score_tmp.max().detach()

            """ pseudo action point selection """
            #右边
            for j in range(act_idx[i] + 1, act_idx[i+1]): # 两个point实例之间
                if bkg_score[b][j] <= torch.max(cas_sigmoid_fuse[b][j][:-1]) and bkg_seed[b][j] < 1: # 背景点小于point实例的分数
                    act_seed[b, j] = cas_sigmoid_fuse[b, j]
                    # act_anno[b, j] = act_anno[b, act_idx[i]]
                else:
                    break
            #左边
            for j in range(act_idx[i+1] - 1, act_idx[i], -1):
                if bkg_score[b][j] <= torch.max(cas_sigmoid_fuse[b][j][:-1]) and bkg_seed[b][j] < 1:
                    act_seed[b, j] = cas_sigmoid_fuse[b, j]
                    # act_anno[b, j] = act_anno[b, act_idx[i+1]]
                else:
                    break
    return act_seed, bkg_seed
def gaussian_smoothing(S, r=0.25):
   
    B, T, C = S.shape
    t = torch.arange(T, device=S.device).float()
    kernel = torch.exp(-((t[:, None] - t[None, :]) ** 2) / (2 * r ** 2)) / torch.sqrt(2 * torch.tensor(3.1415926, device=S.device) * r ** 2)
    kernel = kernel.clamp(min=1e-5)
    kernel = kernel / kernel.sum(dim=1, keepdim=True)  # 归一化
    smoothed_S = torch.zeros_like(S)
    for b in range(B):
        for c in range(C):
            smoothed_S[b, :, c] = torch.matmul(kernel, S[b, :, c].unsqueeze(-1)).squeeze(-1)
    smoothed_S = smoothed_S.clamp(min=1e-5)
    return smoothed_S
def BSPG(point_labels, step_width=3, peak_value=1, base_value=0.7):
    """
    Parameters:
        point_labels: Annotation tensor of shape [B,T,C], where 0.9 indicates positive samples, 0.1 indicates negative samples/suppression items, and 0.0 indicates no annotation
        step_width: Width to extend from each annotation point to both sides
        peak_value: Peak weight at the annotation point
        base_value: Base weight for the extended region
    """
    B, T, C = point_labels.shape
    device = point_labels.device
    step_labels = torch.zeros((B, T, C+1), device=device)  
    
    
    step_labels[..., -1] = 0.0
    
    for b in range(B):
       
        strong_positions = torch.nonzero(point_labels[b] == 0.9)
        weak_positions = torch.nonzero(point_labels[b] == 0.1)
        
       
        for t, cls_idx in strong_positions:
            start = max(0, t - step_width)
            end = min(T, t + step_width + 1)
            
          
            weights = torch.linspace(base_value, peak_value, step_width + 1, device=device)
            left_weights = weights[:t - start + 1]
            right_weights = weights.flip(0)[:end - t]
            full_weights = torch.cat([left_weights[:-1], right_weights])
            
         
            step_labels[b, start:end, cls_idx] = torch.maximum(
                step_labels[b, start:end, cls_idx],
                full_weights
            )
        
     
        for t, cls_idx in weak_positions:
            step_labels[b, t, cls_idx] = base_value
        
        foreground = torch.max(step_labels[b, :, :-1], dim=-1)[0]
        step_labels[b, :, -1] = 1.0 - foreground
    
    return step_labels