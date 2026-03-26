
import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
import numpy as np
import math
from scipy import ndimage
import os
def compute_frame_precision(gtis, pred):
    device = pred.device
    gt_labels = torch.argmax(gtis.to(device), dim=-1)     # (B, T)
    pred_labels = torch.argmax(pred, dim=-1)              # (B, T)
    bg_class = pred.shape[-1] - 1 
    valid_mask = (gt_labels != bg_class) & (pred_labels != bg_class)
    tp = ((pred_labels == gt_labels) & valid_mask).sum().item()
    pred_positive = (pred_labels != bg_class).sum().item()
    gt_positive = (gt_labels != bg_class).sum().item()

    return tp, pred_positive, gt_positive

class SniCoLoss(nn.Module):
    def __init__(self):
        super(SniCoLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()

    def NCE(self, q, k, neg, T=0.07):
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)

        return loss

    def forward(self, contrast_pairs):

        HA_refinement = self.NCE(
            torch.mean(contrast_pairs['HA'], 1), 
            torch.mean(contrast_pairs['EA'], 1), 
            contrast_pairs['EB']
        )

        HB_refinement = self.NCE(
            torch.mean(contrast_pairs['HB'], 1), 
            torch.mean(contrast_pairs['EB'], 1), 
            contrast_pairs['EA']
        )

        loss = HA_refinement + HB_refinement
        return loss


class GeneralizedCE(nn.Module):
    def __init__(self, q=0.07):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, cas,point):
        # cas shape: B T C
        # logits shape: B T
        #point B T C
        point=point.max(dim=-1)[0]
        k = cas.shape[1] //8
        topk_values, topk_indices = torch.topk(cas, k, dim=1)
        
        label = torch.zeros_like(cas, dtype=torch.float)
        for i in range(cas.shape[0]):
            for j in range(cas.shape[2]):
                label[i, topk_indices[i, :,j], j] = 1
        label = label.sum(dim=-1) > 0
        label = label.float()
        label=((label+point)>0).float()
        pos_factor = torch.sum(label, dim=1) + 1e-7
        neg_factor = torch.sum(1 - label, dim=1) + 1e-7
        label = label.to(logits.device)
        pos_factor = pos_factor.to(logits.device)
        neg_factor = neg_factor.to(logits.device)
        first_term = torch.mean(torch.sum(((1 - (logits + 1e-7)**self.q)/self.q) * label, dim=1)/pos_factor)
        second_term = torch.mean(torch.sum(((1 - (1 - logits + 1e-7)**self.q)/self.q) * (1-label), dim=1)/neg_factor)
        
        return first_term + second_term

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
        self.ce_criterion = nn.CrossEntropyLoss()
        self.k_hard=8
        self.k_easy=32
        self.m = 10
        self.M = 20
        self.dropout = nn.Dropout(p=0.5)

    def cosine_similarity_matrix(self,features):
        """
        Calculate the cosine similarity matrix between feature vectors for each batch.
        features: B x T x D (B=batch size, T=sequence length, D=feature dimension)
        """
        features = F.normalize(features, p=2, dim=2)
        similarity_matrix = torch.bmm(features, features.transpose(1, 2))
        return similarity_matrix
    def NCE(self, q, k, neg, T=0.1):                #　　0.1
        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)
        neg = neg.permute(0,2,1)
        neg = nn.functional.normalize(neg, dim=1)
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,nck->nk', [q, neg])
        logits = torch.cat([l_pos, l_neg], dim=1)
        logits /= T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
        loss = self.ce_criterion(logits, labels)
        return loss

    def select_topk_embeddings(self, scores, embeddings, k):
        assert scores.dim() == 2, f"scores 的形状应为 (B, T)，但实际是 {scores.shape}"
        _, idx_DESC = scores.sort(descending=True, dim=1)  # idx_DESC 的形状是 (B, T)
        idx_topk = idx_DESC[:, :k]  
        idx_topk = idx_topk.unsqueeze(2)  # idx_topk 的形状变为 (B, k, 1)
        idx_topk = idx_topk.expand(-1, -1, embeddings.size(2))  # idx_topk 的形状变为 (B, k, D)
        selected_embeddings = torch.gather(embeddings, 1, idx_topk)  # 输出形状是 (B, k, D)
        return selected_embeddings
    
    def easy_snippets_mining(self, actionness, embeddings, k_easy,k_hard):
        select_idx = torch.ones_like(actionness).cuda()
        select_idx = self.dropout(select_idx)
        actionness_rev = torch.max(actionness, dim=1, keepdim=True)[0] - actionness
        actionness_rev_drop = actionness_rev * select_idx
        easy_bkg = self.select_topk_embeddings( actionness_rev_drop, embeddings, k_easy)
        # print(actionness.shape)
        aness_np = actionness.cpu().detach().numpy()
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)

        dilation_m = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.m))).astype(aness_np.dtype)
        dilation_M = ndimage.binary_dilation(aness_bin, structure=np.ones((1,self.M))).astype(aness_np.dtype)
        idx_region_outer = actionness.new_tensor(dilation_M - dilation_m)
        aness_region_outer = actionness * idx_region_outer
        hard_bkg = self.select_topk_embeddings(aness_region_outer, embeddings, k_hard)

        return easy_bkg, hard_bkg

    def Inconsistency_snippets_mining1(self,actionness, embeddings,neighbor,act_seed, k_hard=20):
        B,T ,C=act_seed.shape
        # print(act_seed.shape)
        k_hard=T//self.k_hard
        k_easy=T//self.k_easy
        # print(k_easy,k_hard)
        aness_np = actionness.cpu().detach().numpy()  # i+T
        aness_median = np.median(aness_np, 1, keepdims=True)
        aness_bin = np.where(aness_np > aness_median, 1.0, 0.0)
        aness_bin = torch.from_numpy(aness_bin).float()
        aness_bin = aness_bin.to(actionness.device)
        if not isinstance(act_seed, torch.Tensor):
            act_seed = torch.from_numpy(act_seed).float()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备
        act_seed = act_seed.to(device)
        aness_bin = aness_bin.to(device)
    
        aness_np = aness_bin * act_seed #B T C
        sampel_np=actionness*aness_np
        aness_np_expanded = aness_np.unsqueeze(2)  # Shape: [B, T, 1, C]
        neighbor_expanded = neighbor.unsqueeze(0).unsqueeze(-1)  # Shape: [1, T, T, 1]
        neighborF = aness_np_expanded * neighbor_expanded  # Shape: [B, T, T, C]
        # print(neighborF.shape)
        neighborF = torch.sum(neighborF, dim=2)  # Shape: [B, T, C]
        neighborF=neighborF.squeeze(1)
        torch.set_printoptions(threshold=float('inf'))
        pk=self.select_topk_embeddings(sampel_np[:,:,0],embeddings, k_easy)
        nk=self.select_topk_embeddings(sampel_np[:,:,1],embeddings, k_easy)
        neighbor_pk = self.select_topk_embeddings(neighborF[:,:,0], embeddings, k_hard)
        neighbor_nk=self.select_topk_embeddings(neighborF[:,:,1], embeddings, k_hard)
        bkg,neighbor_bkg=self.easy_snippets_mining(1-actionness.squeeze(-1),embeddings,k_easy,k_hard)
        # print(pk.shape,nk.shape,neighbor_pk.shape,neighbor_nk.shape,bkg.shape,neighbor_bkg.shape)
        contrast_pairs={
            "S_p":pk,
            "S_n":nk,
            "N_p":neighbor_pk,
            "N_n":neighbor_nk,
            "S_b":bkg,
            "N_b":neighbor_bkg}
        return contrast_pairs
    

    def forward(self, actionness,feature,point_lable):
        similarity_matrix=self.cosine_similarity_matrix(feature)
        contrast_pairs=self.Inconsistency_snippets_mining1(actionness,feature,similarity_matrix,point_lable,k_hard=20)
        def flatten_and_mean(tensor):
            B, k, D = tensor.shape
            return tensor.reshape(B * k, D)

        S_p = contrast_pairs['S_p']
        S_n = contrast_pairs['S_n']
        N_p = contrast_pairs['N_p']
        N_n = contrast_pairs['N_n']
        S_b = contrast_pairs['S_b']
        N_b = contrast_pairs['N_b']
        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['N_p'], 1),
            torch.mean(contrast_pairs['S_p'], 1),
            contrast_pairs['N_n'])
        # +self.NCE(
        #     torch.mean(contrast_pairs['N_p'], 1),
        #     torch.mean(contrast_pairs['S_p'], 1),
        #     contrast_pairs['N_b']
        # )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['N_n'], 1),
            torch.mean(contrast_pairs['S_n'], 1),
            contrast_pairs['N_p'])
        # +self.NCE(
        #     torch.mean(contrast_pairs['N_p'], 1),
        #     torch.mean(contrast_pairs['S_p'], 1),
        #     contrast_pairs['N_b']
        # )
        loss =0.5*IA_refinement + 0.5*IB_refinement
        return loss

class Total_loss(nn.Module):
    def __init__(self, lambdas):
        super(Total_loss, self).__init__()
        self.tau = 0.1
        self.sampling_size = 3
        self.lambdas = lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')
        self.frame_ldl_criterion = nn.KLDivLoss(reduction='none')
        self.ldl_criterion = nn.KLDivLoss()
        self.cl=SniCoLoss()
        self.sent=GeneralizedCE()
        self.feat=ContrastiveLoss()

    def forward(self, vid_score, cas_sigmoid_fuse, cas_softmax_dis, vid_distribution, features, sentiment,stored_info, _label_distribution, label, point_anno, cpc_loss, step):
        # self, vid_score, cas_sigmoid_fuse, cas_softmax_dis,  features, cas_aug,cas_sentiment, stored_info, _label_distribution, label, point_anno, cpc_loss, step
        loss = {}
        print(vid_distribution.shape,_label_distribution.shape)

        loss_vid_ldl = self.ldl_criterion(torch.log(vid_distribution), _label_distribution).mean()     
        # loss_cl=self.cl(contrast_pairs)
        loss_sentiment=self.Lgce(sentiment,cas_sigmoid_fuse,point_anno)
        print(loss_sentiment)
        loss_CL=self.CL(sentiment,features,point_anno)
        print(loss_CL)
        loss_senti=self.sent(sentiment.squeeze(-1),cas_sigmoid_fuse[:,:,:-1],point_anno)
        loss_cl=self.feat(sentiment,features,point_anno)

        step_point=utils.create_multi_step_labels(point_anno.detach().cpu(), step_width=5, peak_value=0.9, base_value=0.1)
        point_anno = torch.cat((point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1)).cuda()), dim=2)
        # loss_feat=self.feat(sentiment,features,point_anno)

        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1)

        is_background = (_full_label.sum(dim=2) == 0).unsqueeze(-1)  # shape: (B, T, 1)
        _full_label = torch.cat([_full_label, is_background.float()], dim=2)
        # point_anno=step_point
        # point_anno=point_anno.cuda()
        focal_weight_act = (1 - cas_sigmoid_fuse) * point_anno+ cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2
        # ce
        loss_frame = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()

        # soft bkg
        act_seed, bkg_seed = utils.select_seed_act_score(cas_sigmoid_fuse.detach().cpu(), point_anno.detach().cpu())
        pos_num = (act_seed.max(dim=2)[0]>0).int().sum(dim=1).cuda()
        neg_num = ((bkg_seed>0).int().sum()).cuda()
        tot_num = pos_num+neg_num+1e-5
        rate = pos_num / neg_num

        # act_seed=torch.clamp(act_seed+step_point, min=0.0, max=1.0)
        act_seed = act_seed.cuda()

                
        # soft act
        weighting_p_act = act_seed
        num_p_actions = act_seed.max(dim=2)[0].sum(dim=1)
        
        if num_p_actions>0:
            focal_weight_p_act = (1 - cas_sigmoid_fuse) * (act_seed>0.5).int() + cas_sigmoid_fuse * (act_seed<0.5).int()
            focal_weight_p_act = focal_weight_p_act ** 2
            loss_frame_pact = 0.5 * (((focal_weight_p_act * self.ce_criterion(cas_sigmoid_fuse, act_seed) * weighting_p_act).sum(dim=2)).sum(dim=1) / num_p_actions).mean()
        else:
            loss_frame_pact = torch.zeros(1)[0].cuda()
            
        bkg_seed = bkg_seed.unsqueeze(-1).cuda()

        point_anno_bkg = torch.zeros_like(point_anno).cuda()
        point_anno_bkg[:,:,-1] = 1

        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1)

        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2
        # ce 
        loss_frame_bkg = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean()
        loss_total =  self.lambdas[0] * loss_vid_ldl+ self.lambdas[1] * ((1-self.lambdas[2]) * loss_frame + self.lambdas[2] * (loss_frame_bkg + loss_frame_pact)) + self.lambdas[3] * cpc_loss+loss_cl+loss_senti
        loss['loss_recover_cpc'] = cpc_loss
        loss["loss_vid_ldl"] = loss_vid_ldl
        loss["loss_frame"] = loss_frame
        loss["loss_frame_bkg"] = loss_frame_bkg
        loss["loss_frame_pact"] = loss_frame_pact        
        loss["loss_total"] = loss_total
        loss["pos_neg_rage"] = rate
        loss["pos_num"] = pos_num
        loss["neg_num"] = neg_num
        return loss_total, loss



def train_all(net, config, loader_iter, optimizer, criterion, logger, step):
    net.train()

    total_loss = {}
    total_cost = []

    optimizer.zero_grad()

    for _b in range(config.batch_size):

        _, _data, _label, _point_anno, stored_info, _, _, _label_distribution = next(loader_iter)
        # stored_info  _label_distribution 
        _data = [_data[0].cuda(),_data[1].cuda(),_data[2].cuda()]
        _label = _label.cuda() #B*C [1,0],[0,1],[1,1]
        _label_distribution = _label_distribution.cuda() #B*C
        _point_anno = _point_anno.cuda() #B*T*C
        
        # Check the actual return values from the model
        model_output = net(_data, _label)
       
        vid_score, cas_sigmoid_fuse, cas_softmax_dis, features,vid_distribution,cas_sentiment,cpc_loss= model_output
        
        cost, loss = criterion(vid_score, cas_sigmoid_fuse, cas_softmax_dis, vid_distribution,features, cas_sentiment, stored_info, _label_distribution, _label, _point_anno, cpc_loss, step)
        if torch.isnan(cost):
            print("Cost is NaN")
            cost = torch.zeros(1, requires_grad=True).cuda()
        
        total_cost.append(cost)

        for key in loss.keys():
            if not (key in total_loss):
                total_loss[key] = []

            if isinstance(loss[key], torch.Tensor) and loss[key] > 0:
                total_loss[key] += [loss[key].detach().cpu().item()]
            else:
                total_loss[key] += [loss[key]]

    total_cost = sum(total_cost) / config.batch_size

    # Check for NaN before backward
    if not torch.isnan(total_cost):
        total_cost.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=5.0)
        total_norm = 0
        for p in net.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        # print("Gradient Norm:", total_norm)
        optimizer.step()
    else:
        print("Skipping backward pass due to NaN cost")

    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / config.batch_size, step)
