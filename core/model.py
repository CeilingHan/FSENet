import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import numpy as np
from scipy import ndimage
from transformer import FSD

def positionalencoding1d(d_model, length):
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

def time_mesh(T, device):
    x = torch.arange(T).view(1,T).repeat(T,1)
    y = torch.arange(T).view(T,1).repeat(1,T)
    
    meshs = 0.5+(torch.abs(x-y)/T).to(device)
    return meshs

class CPC(nn.Module):
    """
        Contrastive Predictive Coding: score computation. See https://arxiv.org/pdf/1807.03748.pdf.

        Args:
            x_size (int): embedding size of input modality representation x
            y_size (int): embedding size of input modality representation y
    """
    def __init__(self, x_size, y_size, num_layer=4, activation='ReLU'):
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.activation = getattr(nn, activation)
        map=[]
        for _ in range(num_layer-1):
            map.append(nn.Conv1d(in_channels=self.y_size, out_channels=self.y_size, kernel_size=3,
                      stride=1, padding=1))
            map.append(self.activation())
        map.append(nn.Conv1d(in_channels=self.y_size, out_channels=self.x_size, kernel_size=3,
                      stride=1, padding=1))
        map.append(self.activation())
        self.net = nn.Sequential(*map)
        
    # Ours
    def forward(self, x, y):
        """Calulate the score 
        """
        # x = x
        # y = y
        T = x.size(1)
        # tmesh = time_mesh(T, x.device)
        x_pred = torch.flatten(self.net(y).permute(0,2,1),start_dim=0,end_dim=1).contiguous()    # bs x T, emb_size
        x = torch.flatten(x,start_dim=0,end_dim=1).contiguous() # bs x T, emb_size

        # normalize to unit sphere
        x_pred = x_pred / x_pred.norm(dim=-1, keepdim=True)
        x = x / x.norm(dim=-1, keepdim=True)

        pos = torch.sum(x*x_pred, dim=-1)   # bs
        neg = torch.logsumexp(torch.matmul(x, x_pred.t()), dim=-1)   # bs
        nce = (pos - neg).mean()
        return nce
    

        

class FSENet(nn.Module):
    def __init__(self, len_feature, num_classes):
        super(FSENet, self).__init__()
        self.len_feature = len_feature
        
        # temporal embedding
        self.tpe = positionalencoding1d(60,4000).unsqueeze(0).unsqueeze(-2)
        
        # audio branch ResNet18
        # a_resnet = torchvision.models.resnet18(pretrained=True)
        # a_conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2), bias=False)
        # a_pool = nn.AvgPool2d(kernel_size=[1, 2])
        # a_res = [a_conv1] + list(a_resnet.children())[1:-2] + [a_pool]
        
        # audio branch 3-layer CNN
        a_l1 = [nn.Conv2d(1, 64, kernel_size=[7, 7], stride=(3, 2), padding=(3, 3)),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=[4, 4], stride=[4, 4]),
                ]
        a_l2 = [
            nn.Conv2d(64, 256, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=[1, 1], stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=[2, 2], stride=[2, 2])
        ]
        a_l3 = [
            nn.Conv2d(256, 512, kernel_size=[3, 3], stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=[1, 1], stride=(1, 1)),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.AvgPool2d(kernel_size=[4, 3])
        ]
        
        a_res = a_l1 + a_l2 + a_l3
        self.a_extractor = nn.Sequential(*a_res)
        
        # feature align
        self.v_align = nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU())
        #audio 
        self.a_align = nn.Sequential(
        nn.Conv1d(in_channels=768, out_channels=512, kernel_size=3,
                    stride=1, padding=1),
        nn.ReLU())

        self.f_align=nn.Sequential(
            nn.Conv1d(in_channels=1024, out_channels=512, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU())
        
        # fuse conv
        self.neck = nn.Sequential(
            nn.Conv1d(in_channels=512*3, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )
        self.neck_mix = nn.Sequential(
            nn.Conv1d(in_channels=self.len_feature, out_channels=2048, kernel_size=3,
                      stride=1, padding=1),
            nn.ReLU()
        )

        self.mulfuse=FSD()
        self.rev_fa = CPC(512,2048)
        self.rev_fv = CPC(512,2048)
        self.rev_ff=CPC(512,2048)
        
        self.classifier = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes+1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.distribution = nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=num_classes+1, kernel_size=1,
                      stride=1, padding=0, bias=False)
        )
        self.senti=nn.Sequential(
            nn.Conv1d(in_channels=2048, out_channels=1, kernel_size=1,
                      stride=1, padding=0, bias=True)
        )
        self.drop_out = nn.Dropout(p=0.7)
        self.D=2048
        self.gamma =0.66
   
        
    def forward(self, x, istrain):
        cpc_loss = None
        v_fea, a_fea,f_fea = x
        # print(f_fea.shape)
        # if use Senti video logmfcc feature a_fea is B,T,H,W，else audio is B,T,768
        # B,T,H,W = a_fea.shape
        # a_fes = []
        # for t in range(0,T,600):
        #     tlen = min(600,T-t)
        #     a_fe = a_fea[:,t:t+tlen]+self.tpe[:,t:t+tlen].to(v_fea.device)
        #     a_fe = self.a_extractor(torch.cat([a_fe.roll(1,1).view(B*tlen,1,H,W), a_fe.view(B*tlen,1,H,W), a_fe.roll(-1,1).view(B*tlen,1,H,W)],dim=2))
        #     a_fe = torch.flatten(a_fe, start_dim=1).contiguous().view(B,tlen,512)
        #     a_fes.append(a_fe)
        # a_fea = torch.cat(a_fes,dim=1)
        # del a_fe,a_fes
        # torch.cuda.empty_cache()
        a_fea=self.a_align(a_fea.permute(0,2,1)).permute(0,2,1)
        v_fea = self.v_align(v_fea.permute(0,2,1)).permute(0,2,1)
        f_fea = self.f_align(f_fea.permute(0,2,1)).permute(0,2,1)
        mixtrue=self.mulfuse(a_fea,v_fea,f_fea)#B,T,D 1024
      
        x = torch.cat([a_fea,v_fea,f_fea],dim=-1)
        out = x.permute(0, 2, 1)
        out = self.neck(out)
        mixtrue=self.neck_mix(mixtrue.permute(0, 2, 1))
        # CPC
        if istrain:
            cpc_fa = self.rev_fa(a_fea, out)
            cpc_fv = self.rev_fv(v_fea, out)
            cpc_ff = self.rev_ff(f_fea, out)
            cpc_loss = cpc_fa + cpc_fv + cpc_ff
        feat = out.permute(0, 2, 1)
        if torch.isnan(feat).any():
            print("NaN detected in feat")
        if torch.isinf(feat).any():
            print("Inf detected in feat")
        out = self.drop_out(out)
        cas = self.classifier(mixtrue)
        # cas_aug=self.classifier(mixtrue).permute(0, 2, 1)
        sentiness=self.senti(out).permute(0, 2, 1)
        cas_dis = self.distribution(mixtrue)
        cas = cas.permute(0, 2, 1)
        cas_dis = cas_dis.permute(0, 2, 1)
        return feat, cas, cas_dis,sentiness, cpc_loss

        
class Model(nn.Module):
    def __init__(self, len_feature, num_classes, r_act=8):
        super(Model, self).__init__()
        self.len_feature = len_feature
        self.num_classes = num_classes
        self.r_act =r_act # topk, set as top 1/8 of sequence
        self.cls_module = FSENet(len_feature, num_classes)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)        

    def forward(self, x, vid_labels=None):
        istrain = not vid_labels is None
        # x: Batch x Time x Channel
        num_segments = x[0].shape[1]
        k_act = num_segments // self.r_act
        features, cas, cas_dis,sentiness, cpc_loss = self.cls_module(x, istrain)

        # cas_act = cas[:, :, :-1]
       
        sentiness=torch.sigmoid(sentiness)
        contrast_pairs = {
        }
        
        cas_sigmoid = self.sigmoid(cas)
        # C class * foreground/background score
        cas_sigmoid_fuse = cas_sigmoid[:,:,:-1] * (1 - cas_sigmoid[:,:,-1].unsqueeze(2))
        # overall score cat with fg score
        cas_sigmoid_fuse = torch.cat((cas_sigmoid_fuse, cas_sigmoid[:,:,-1].unsqueeze(2)), dim=2)
        
        dis_topk, _ = cas_dis[:,:,:-1].sort(descending=True, dim=1)
        dis_topk = dis_topk[:,:k_act]
        
        value, _ = cas_sigmoid.sort(descending=True, dim=1)
        topk_scores = value[:,:k_act,:-1] # B topk C

        if vid_labels is None:
            vid_score = torch.mean(topk_scores, dim=1) # B C
            return vid_score, cas_sigmoid_fuse, features,sentiness
        else:
            vid_ldl = torch.softmax(torch.mean(dis_topk, dim=1), dim=1)
            vid_score = (torch.mean(topk_scores, dim=1) * vid_labels) + (torch.mean(cas_sigmoid[:,:,:-1], dim=1) * (1 - vid_labels))
            cas_softmax_dis = torch.softmax(cas_dis, dim=2)
            
            return vid_score, cas_sigmoid_fuse, cas_softmax_dis, features, vid_ldl, sentiness,cpc_loss,contrast_pairs
