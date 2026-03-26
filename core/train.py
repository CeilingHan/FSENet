import torch
import torch.nn as nn
import utils
import torch.nn.functional as F
import numpy as np
from scipy import ndimage

class SoftMatch(nn.Module):
    def __init__(self, num_classes=3, T=0.5, n_sigma=2, momentum=0.999, per_class=False, lambda_u=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.T = T  # 温度参数
        self.lambda_u = lambda_u  # 无监督损失权重
        self.n_sigma = n_sigma
        self.momentum = momentum
        self.per_class = per_class

        # 初始化高斯分布参数
        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes)
            self.prob_max_var_t = torch.tensor(1.0)
        else:
            self.prob_max_mu_t = torch.ones((self.num_classes)) / self.num_classes
            self.prob_max_var_t = torch.ones((self.num_classes))

    @torch.no_grad()
    def update_gaussian_stats(self, probs_x_ulb):
        """更新高斯分布统计信息"""
        max_probs, max_idx = probs_x_ulb.max(dim=-1)
        
        if not self.per_class:
            prob_max_mu_t = torch.mean(max_probs)
            prob_max_var_t = torch.var(max_probs, unbiased=True)
            self.prob_max_mu_t = self.momentum * self.prob_max_mu_t + (1 - self.momentum) * prob_max_mu_t.item()
            self.prob_max_var_t = self.momentum * self.prob_max_var_t + (1 - self.momentum) * prob_max_var_t.item()
        else:
            prob_max_mu_t = torch.zeros_like(self.prob_max_mu_t)
            prob_max_var_t = torch.ones_like(self.prob_max_var_t)
            for i in range(self.num_classes):
                prob = max_probs[max_idx == i]
                if len(prob) > 1:
                    prob_max_mu_t[i] = torch.mean(prob)
                    prob_max_var_t[i] = torch.var(prob, unbiased=True)
            self.prob_max_mu_t = self.momentum * self.prob_max_mu_t + (1 - self.momentum) * prob_max_mu_t
            self.prob_max_var_t = self.momentum * self.prob_max_var_t + (1 - self.momentum) * prob_max_var_t
        
        return max_probs, max_idx

    def calculate_mask(self, probs_x_ulb):
        """计算置信度掩码"""
        if not self.prob_max_mu_t.is_cuda and probs_x_ulb.is_cuda:
            self.prob_max_mu_t = self.prob_max_mu_t.to(probs_x_ulb.device)
            self.prob_max_var_t = self.prob_max_var_t.to(probs_x_ulb.device)

        max_probs, max_idx = self.update_gaussian_stats(probs_x_ulb)

        if not self.per_class:
            mu = self.prob_max_mu_t
            var = self.prob_max_var_t
        else:
            mu = self.prob_max_mu_t[max_idx]
            var = self.prob_max_var_t[max_idx]

        mask = torch.exp(-((torch.clamp(max_probs - mu, max=0.0) ** 2) / (2 * var / (self.n_sigma ** 2))))
        return mask

    def forward(self, logits_all, y, logits_ulb_s):
        """
        前向传播
        Args:
            logits_all: 所有数据的预测结果 (B, T, num_classes)
            y: 标签 (B, T, C)，1表示有标签，0表示无标签
            logits_ulb_s: 增强后所有数据的预测结果 无标签数据y得到索引
            model: 基础模型
        """
        batch_size, seq_len, num_classes = logits_all.shape
        # 分离有标签和无标签数据
        labeled_mask = (y.sum(dim=-1) > 0)  # (B, T)
        unlabeled_mask = ~labeled_mask  # (B, T)
        
        # 计算有监督损失
        logits_lb = logits_all[labeled_mask]
        y_lb = y[labeled_mask].argmax(dim=-1)  # 获取真实标签的索引
        sup_loss = F.cross_entropy(logits_lb, y_lb, reduction='mean') if len(y_lb) > 0 else torch.tensor(0.0).to(logits_all.device)
        
        # 无标签数据处理
        if unlabeled_mask.any():
            with torch.no_grad():
                logits_ulb_w = logits_all[unlabeled_mask]  # (N_ulb, num_classes)
                probs_ulb_w = torch.softmax(logits_ulb_w / self.T, dim=-1)
            
            # 确保logits_ulb_s与logits_all具有相同的形状
            if logits_ulb_s.shape != logits_all.shape:
                logits_ulb_s_filtered = logits_ulb_s
            else:
                # 如果形状匹配，则使用相同的掩码进行索引
                logits_ulb_s_filtered = logits_ulb_s[unlabeled_mask]
            
            # 形状检查
            N_ulb = unlabeled_mask.sum()
            
            pseudo_labels = torch.argmax(probs_ulb_w, dim=-1)
            
            mask = self.calculate_mask(probs_ulb_w)
            
            # 计算无监督损失
            per_sample_loss = F.cross_entropy(logits_ulb_s_filtered, pseudo_labels, reduction='none')  # (N_ulb,)
            unsup_loss = (per_sample_loss * mask).mean()
        else:
            unsup_loss = torch.tensor(0.0).to(logits_all.device)
        
        # 总损失
        total_loss = sup_loss + self.lambda_u * unsup_loss
        
        return total_loss, sup_loss, unsup_loss
class Focal(nn.Module):
    def __init__(self, lambdas): 
        super(Focal, self).__init__()
        self.tau = 0.1
        self.sampling_size = 3
        self.lambdas = lambdas
        self.ce_criterion = nn.BCELoss(reduction='none')
        self.frame_ldl_criterion = nn.KLDivLoss(reduction='none')
        self.eps = 1e-8  # divide by zero

    def forward(self, cas_sigmoid_fuse, point_anno):
        act_seed1 = utils.BSPG(point_anno)
        point_anno = torch.cat((
            point_anno, 
            torch.zeros((point_anno.shape[0], point_anno.shape[1], 1), device=point_anno.device)
        ), dim=2)
        
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1) + self.eps  
        cas_sigmoid_fuse = torch.clamp(cas_sigmoid_fuse, min=1e-7, max=1-1e-7)
        # focal loss
        focal_weight_act = (1 - cas_sigmoid_fuse) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2
        loss_frame = (((focal_weight_act * self.ce_criterion(cas_sigmoid_fuse, point_anno) * weighting_seq_act).sum(dim=2)).sum(dim=1) / num_actions).mean()
        act_seed, bkg_seed = utils.select_seed_act_score(
            cas_sigmoid_fuse.detach().cpu(), 
            point_anno.detach().cpu()
        )
        act_seed1 = utils.BSPG(act_seed)
        pos_num = (act_seed.max(dim=2)[0] > 0).int().sum(dim=1).to(point_anno.device)
        neg_num = ((bkg_seed > 0).int().sum()).to(point_anno.device)
        rate = pos_num / (neg_num + self.eps)  
        act_seed = act_seed.to(point_anno.device).clamp(0, 1)
        weighting_p_act = act_seed
        num_p_actions = act_seed.max(dim=2)[0].sum(dim=1) + self.eps  
        if num_p_actions.sum() > self.eps:
            focal_weight_p_act = (1 - cas_sigmoid_fuse) * (act_seed > 0.5).int() + cas_sigmoid_fuse * (act_seed < 0.5).int()
            focal_weight_p_act = focal_weight_p_act ** 2
            loss_frame_pact = 0.5 * (((focal_weight_p_act * self.ce_criterion(cas_sigmoid_fuse, act_seed) * weighting_p_act).sum(dim=2)).sum(dim=1) / num_p_actions).mean()
        else:
            loss_frame_pact = torch.tensor(0.0, device=point_anno.device)
        bkg_seed = bkg_seed.unsqueeze(-1).to(point_anno.device)
        point_anno_bkg = torch.zeros_like(point_anno, device=point_anno.device)
        point_anno_bkg[:, :, -1] = 1 
        
        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1) + self.eps  
        
        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2
        loss_frame_bkg = (((focal_weight_bkg * self.ce_criterion(cas_sigmoid_fuse, point_anno_bkg) * weighting_seq_bkg).sum(dim=2)).sum(dim=1) / num_bkg).mean()
        loss_frame_total = self.lambdas[1] * ((1 - self.lambdas[2]) * loss_frame + self.lambdas[2] * (loss_frame_bkg + loss_frame_pact))
        return loss_frame_total, loss_frame, loss_frame_bkg, loss_frame_pact, rate, pos_num, neg_num
class GeneralizedCE(nn.Module):
    def __init__(self, q=0.07):
        self.q = q
        super(GeneralizedCE, self).__init__()

    def forward(self, logits, cas,point):
        # cas shape: B T C
        # logits shape: B T
        #point B T C
        point=point.max(dim=-1)[0]
        k = cas.shape[1] //8 # 例如，top-k 为时间步长的 1/32
        topk_values, topk_indices = torch.topk(cas, k, dim=1)  # 获取每个时间步的每个类别的 top-k 激活值和对应的索引
        
        label = torch.zeros_like(cas, dtype=torch.float)
        for i in range(cas.shape[0]):  # 遍历批次
            for j in range(cas.shape[2]):  # 遍历类别
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
        self.normalize = True

    def cosine_similarity_matrix(self, features, metric='cosine'):
        """
        Compute similarity matrix using different metrics.
        Args:
            features: Tensor (B, T, D)
            metric: str, one of ['cosine', 'l1', 'l2', 'dot']
        Returns:
            similarity matrix: Tensor (B, T, T)
        """
        if metric == 'cosine':
            features = F.normalize(features, p=2, dim=2)
            return torch.bmm(features, features.transpose(1, 2))

        elif metric == 'dot':
            return torch.bmm(features, features.transpose(1, 2))

        elif metric == 'l2':
            xx = (features ** 2).sum(dim=2, keepdim=True)
            yy = xx.transpose(1, 2)
            dist = xx + yy - 2 * torch.bmm(features, features.transpose(1, 2))
            dist = torch.clamp(dist, min=1e-6)
            return -dist

        elif metric == 'l1':
            dist = torch.abs(features.unsqueeze(2) - features.unsqueeze(1)).sum(dim=-1)
            return -dist
        elif metric == 'mse':
            B, T, D = features.shape
            result = torch.zeros(B, T, T, device=features.device)
            chunk_size = 32  # Adjust based on your memory constraints
            
            for i in range(0, T, chunk_size):
                for j in range(0, T, chunk_size):
                    chunk_i = features[:, i:i+chunk_size]
                    chunk_j = features[:, j:j+chunk_size]
                    diff = chunk_i.unsqueeze(2) - chunk_j.unsqueeze(1)
                    result[:, i:i+chunk_size, j:j+chunk_size] = -(diff ** 2).sum(dim=-1)
            return result

        else:
            raise ValueError(f"Unsupported similarity metric: {metric}")

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
    

    def NCE1(self, query, key, negatives,temperature=0.07):
        """
        参数:
            query: [B, D] 或 [B, K_q, D] 
            key: [B, K_k, D] - 正样本特征
            negatives: [B, K_n, D] - 负样本特征
        返回:
            loss: 标量值，NCE损失
        """
        # 确保query为二维张量 [B, D]
        if query.dim() == 3:  # [B, K_q, D] -> [B, D]
            query = torch.mean(query, dim=1)  # 沿K_q维度平均
            
        # 特征归一化（如果需要）
        if self.normalize:
            query = F.normalize(query, p=2, dim=1)  # [B, D]
            key = F.normalize(key, p=2, dim=2)      # [B, K_k, D]
            negatives = F.normalize(negatives, p=2, dim=2)  # [B, K_n, D]
        
        # 计算正样本相似度
        # [B, D] × [B, K_k, D] -> [B, K_k]
        pos_sim = torch.bmm(key, query.unsqueeze(2)).squeeze(2) / temperature
        
        # 计算负样本相似度
        # [B, D] × [B, K_n, D] -> [B, K_n]
        neg_sim = torch.bmm(negatives, query.unsqueeze(2)).squeeze(2) / temperature
        
        # 构建对比学习的logits矩阵 [B, K_k + K_n]
        logits = torch.cat([pos_sim, neg_sim], dim=1)
        
        # 构建标签（所有正样本位置为1）
        # 注：这里假设第一个正样本为目标，其他正样本为辅助（根据实际需求调整）
        labels = torch.zeros(logits.size(0), dtype=torch.long).to(query.device)
        
        # 计算交叉熵损失
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
        # print(sampel_np[:,1:6,:],"pk",sampel_np)
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
            # "S_b":bkg,
            # "N_b":neighbor_bkg
            }
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
        print(S_p.shape,S_n.shape,N_p.shape,N_n.shape)
        # S_b = contrast_pairs['S_b']
        # N_b = contrast_pairs['N_b']
        # temp=self.NCE1(
        #     torch.mean(contrast_pairs['N_p'], 1),
        #     contrast_pairs['S_p'],
        #     contrast_pairs['N_n'])
        # print(temp,"----")
        IA_refinement = self.NCE(
            torch.mean(contrast_pairs['N_p'], 1),
            torch.mean(contrast_pairs['S_p'], 1),
            contrast_pairs['N_n'])
        # print(IA_refinement,"----")
        # )+self.NCE(
        #     torch.mean(contrast_pairs['N_p'], 1),
        #     torch.mean(contrast_pairs['S_p'], 1),
        #     contrast_pairs['N_b']
        # )

        IB_refinement = self.NCE(
            torch.mean(contrast_pairs['N_n'], 1),
            torch.mean(contrast_pairs['S_n'], 1),
            contrast_pairs['N_p'])
        # )+self.NCE(
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
        self.focal=Focal(lambdas)
        self.Lgce=GeneralizedCE()
        self.CL=ContrastiveLoss()

    def forward(self, vid_score, cas_sigmoid_fuse, cas_softmax_dis, vid_distribution, features,sentiness, stored_info, _label_distribution, label, point_anno, cpc_loss, step):
        loss = {}
        # torch.Size([1, 2]) torch.Size([1, 4745, 1]) torch.Size([1, 4745, 3]) torch.Size([1, 4745, 2])
        # print(vid_distribution.shape,sentiness.shape,cas_sigmoid_fuse.shape,point_anno.shape)
        loss_vid_ldl = self.ldl_criterion(torch.log(vid_distribution), _label_distribution).mean()  
        print("loss_vid_ldl",loss_vid_ldl)
        loss_sentiment=self.Lgce(sentiness,cas_sigmoid_fuse,point_anno)
        print("loss_sentiment",loss_sentiment)
        # print(loss_sentiment)
        loss_CL=self.CL(sentiness,features,point_anno)
        print("loss_CL",loss_CL)
        cas_fg=cas_sigmoid_fuse[:,:,:-1]*sentiness
        cas_bg=cas_sigmoid_fuse[...,[-1]]*(1-sentiness)

        if torch.isnan(cpc_loss):
            cpc_loss=0.0
            print("cpc_loss",cpc_loss)

        cas = torch.cat([cas_fg, cas_bg], dim=-1)  # (B, T, C+1)
        # cas = cas / (cas.sum(dim=-1, keepdim=True) + 1e-8) 
        invalid_mask = (cas_sigmoid_fuse < 0) | (cas_sigmoid_fuse > 1)
        if invalid_mask.any():
            # if the cas_sigmoid_fuse has invalid values
            invalid_values = cas_sigmoid_fuse[invalid_mask]
            invalid_indices = torch.nonzero(invalid_mask, as_tuple=False)
            cas_sigmoid_fuse = torch.clamp(cas_sigmoid_fuse, min=0.0, max=1.0)
        loss_frame_total,loss_frame,loss_frame_bkg,loss_frame_pact,rate,pos_num,neg_num=self.focal(cas_sigmoid_fuse,point_anno)
        loss_frame_total1,loss_frame1,loss_frame_bkg1,loss_frame_pact1,rate1,pos_num1,neg_num1=self.focal(cas,point_anno)
        loss_total = self.lambdas[3] * cpc_loss+ self.lambdas[0] * loss_vid_ldl+loss_frame_total +self.lambdas[1]*loss_sentiment +self.lambdas[2]*loss_CL
        loss['loss_recover_cpc'] = cpc_loss
        loss["loss_vid_ldl"] = loss_vid_ldl
        loss["loss_frame"] = loss_frame
        loss["loss_frame_bkg"] = loss_frame_bkg
        loss["loss_sentiment"]=loss_sentiment  
        loss["loss_total"] = loss_total
        loss["pos_neg_rage"] = rate
        loss["pos_num"] = pos_num
        loss["neg_num"] = neg_num

        for key in loss:
            if isinstance(loss[key], torch.Tensor):
                if torch.isnan(loss[key]).any():
                    print(f"[Warning] Loss '{key}' contains NaN. Resetting to 0.")
                    loss[key] = torch.zeros_like(loss[key])
            elif isinstance(loss[key], (float, int)):
                if loss[key] != loss[key]:  # NaN check for numbers
                    print(f"[Warning] Value '{key}' is NaN. Resetting to 0.0.")
                    loss[key] = 0.0
        return loss_total, loss


def train_all(net, config, loader_iter, optimizer, criterion, logger, step):
    net.train()

    total_loss = {}
    total_cost = []

    optimizer.zero_grad()

    for _b in range(config.batch_size):

        _, _data, _label, _point_anno, stored_info, _, _, _label_distribution = next(loader_iter)
        # if _data
        _data = [_data[0].cuda(),_data[1].cuda(),_data[2].cuda()]
        _label = _label.cuda()
        _label_distribution = _label_distribution.cuda()
        _point_anno = _point_anno.cuda()
        vid_score, cas_sigmoid_fuse, cas_softmax_dis, features, vid_distribution, sentiness,cpc_loss,contrast_pairs = net(_data, _label)

        cost, loss = criterion(vid_score, cas_sigmoid_fuse, cas_softmax_dis, vid_distribution, features,sentiness, stored_info, _label_distribution, _label, _point_anno, cpc_loss, step)

        total_cost.append(cost)

        for key in loss.keys():
            if not (key in total_loss):
                total_loss[key] = []

            if loss[key] > 0:
                total_loss[key] += [loss[key].detach().cpu().item()]
            else:
                total_loss[key] += [loss[key]]

    total_cost = sum(total_cost) / config.batch_size

    total_cost.backward()
    optimizer.step()

    for key in total_loss.keys():
        logger.log_value("loss/" + key, sum(total_loss[key]) / config.batch_size, step)
