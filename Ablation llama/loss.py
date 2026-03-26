
# loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

class TemporalEmotionLoss(nn.Module):
    """
    mixed precision safe version, only depend on logits, label, point_anno.
    """
    def __init__(self, lambdas=[1, 1, 0.5, 0.1]):
        super().__init__()
        # ✅ safe version: directly use BCEWithLogitsLoss
        self.ce_criterion = nn.BCEWithLogitsLoss(reduction='none')
        self.frame_ldl_criterion = nn.KLDivLoss(reduction='none')
        self.ldl_criterion = nn.KLDivLoss()
        self.lambdas = lambdas
        self.tau = 0.1
        self.sampling_size = 3

    def forward(self, logits, label, point_anno):
        """
        logits: [B, T, C+1]
        label: [B, C] 
        point_anno: [B, T, C]
        """
        print("-------------",logits.shape,label.shape,point_anno.shape)
        with torch.no_grad():
            cas_sigmoid_fuse = torch.sigmoid(logits)
        vid_logits = cas_sigmoid_fuse[:, :, :-1].mean(dim=1)  # mean of C classes
        loss_vid = F.binary_cross_entropy_with_logits(vid_logits, label)


        # construct background dimension
        point_anno = torch.cat(
            (point_anno, torch.zeros((point_anno.shape[0], point_anno.shape[1], 1), device=logits.device)),
            dim=2
        )

        # time steps with actions
        weighting_seq_act = point_anno.max(dim=2, keepdim=True)[0]
        num_actions = point_anno.max(dim=2)[0].sum(dim=1) + 1e-5

        # focal-like 
        focal_weight_act = (1 - cas_sigmoid_fuse) * point_anno + cas_sigmoid_fuse * (1 - point_anno)
        focal_weight_act = focal_weight_act ** 2

        loss_frame = (
            (focal_weight_act * self.ce_criterion(logits, point_anno) * weighting_seq_act)
            .sum(dim=2)
            .sum(dim=1)
            / num_actions
        ).mean()

        # soft bkg 
        act_seed, bkg_seed = utils.select_seed_act_score(
            cas_sigmoid_fuse.detach().cpu(), point_anno.detach().cpu()
        )
        act_seed = act_seed.to(logits.device)
        bkg_seed = bkg_seed.unsqueeze(-1).to(logits.device)

        # background mask
        point_anno_bkg = torch.zeros_like(point_anno, device=logits.device)
        point_anno_bkg[:, :, -1] = 1
        weighting_seq_bkg = bkg_seed
        num_bkg = bkg_seed.sum(dim=1) + 1e-5

        focal_weight_bkg = (1 - cas_sigmoid_fuse) * point_anno_bkg + cas_sigmoid_fuse * (1 - point_anno_bkg)
        focal_weight_bkg = focal_weight_bkg ** 2

        loss_frame_bkg = (
            (focal_weight_bkg * self.ce_criterion(logits, point_anno_bkg) * weighting_seq_bkg)
            .sum(dim=2)
            .sum(dim=1)
            / num_bkg
        ).mean()
        # total loss
        loss_total = (
            self.lambdas[0] * loss_vid
            + self.lambdas[1] * ((1 - self.lambdas[2]) * loss_frame + self.lambdas[2] * loss_frame_bkg)
        )
        return loss_total
