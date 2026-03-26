import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, PeftModel
from torch.amp import autocast, GradScaler
import numpy as np
import pandas as pd
import math
from tqdm import tqdm
from loss import TemporalEmotionLoss

# ================= Dataset ==================
class EmotionDatasetPoint(Dataset):
    def __init__(self, feature_paths, point_csv=None, num_classes=2, sampling='uniform', max_frames=64, mode='train'):
        self.feature_path = feature_paths
        self.mode = mode
        self.num_classes = num_classes
        self.sampling = sampling
        self.max_frames = max_frames

        if self.mode == 'train':
            self.point_anno = pd.read_csv(point_csv)
            self.vid_list = self.point_anno['video_id'].unique()
        elif self.mode == 'test':
            rgb_files = os.listdir(self.feature_path[0])
            self.vid_list = [os.path.splitext(f)[0] for f in rgb_files]
        else:
            raise ValueError("mode should be 'train' or 'test'")

    def __len__(self):
        return len(self.vid_list)

    def __getitem__(self, index):
        vid_name = self.vid_list[index]

        rgb_feature_path = os.path.join(self.feature_path[0], vid_name + '.npy')
        mfcc_feature_path = os.path.join(self.feature_path[1], vid_name + '.npy')
        img_feature_path = os.path.join(self.feature_path[2], vid_name + '.npy')

        rgb_feature = np.load(rgb_feature_path, allow_pickle=True).astype(np.float32)
        mfcc_feature = np.load(mfcc_feature_path, allow_pickle=True).astype(np.float32)
        img_feature = np.load(img_feature_path).astype(np.float32)
        # print("=============",rgb_feature.shape,mfcc_feature.shape,img_feature.shape)

        T = min(rgb_feature.shape[0], int(mfcc_feature.shape[0]/32), img_feature.shape[0], self.max_frames)
        rgb_feature = rgb_feature[:T]
        img_feature = img_feature[:T]
        mfcc_feature = mfcc_feature[:T*32].reshape(T,32,60)
        mfcc_feature = (mfcc_feature + 50)/80

        sample_idx = self.uniform_sampling(T) if self.sampling=='uniform' else self.random_perturb(T)
        rgb_feature = rgb_feature[sample_idx]
        img_feature = img_feature[sample_idx]
        mfcc_feature = mfcc_feature[sample_idx]

        if self.mode == 'train':
            temp_anno = np.zeros([T, self.num_classes], dtype=np.float32)
            t_factor = 1/16
            temp_df = self.point_anno[self.point_anno["video_id"]==vid_name][['point','class_index']]
            label = np.zeros([self.num_classes], dtype=np.float32)
            for key in temp_df['point'].keys():
                point = temp_df['point'][key]
                class_idx = temp_df['class_index'][key]
                idx = int(point*t_factor)
                if idx < T:
                    temp_anno[idx, class_idx] = 1
                    label[class_idx] = 1
            point_label = temp_anno[sample_idx,:]

            return {
                "vid": vid_name,
                "rgb": torch.tensor(rgb_feature),
                "mfcc": torch.tensor(mfcc_feature),
                "img": torch.tensor(img_feature),
                "label": torch.tensor(label),
                "point_label": torch.tensor(point_label)
            }
        else:
            return {
                "vid": vid_name,
                "rgb": torch.tensor(rgb_feature),
                "mfcc": torch.tensor(mfcc_feature),
                "img": torch.tensor(img_feature)
            }

    def uniform_sampling(self, length, num_sample_clips=8):
        idx = np.linspace(0, length-1, num_sample_clips).astype(int)
        return idx

    def random_perturb(self,length,num_sample_clips=8):
        idx = np.linspace(0,length-1,num_sample_clips)
        perturb = np.random.randint(-1,2,size=idx.shape)
        idx = np.clip(idx + perturb, 0, length-1).astype(int)
        return idx

# ================= Model helper ==================
def load_llama_lora(model_path, lora_r=64, lora_alpha=16, device="cuda"):
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = "$$"

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model = prepare_model_for_int8_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )
    model.to(device)
    return model, tokenizer

def positionalencoding1d(d_model, length):
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                         -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)
    return pe

class MultiModalAlign(nn.Module):
    def __init__(self, v_dim=1024, a_dim=(32,60), f_dim=256, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.tpe = positionalencoding1d(a_dim[1], 4000).unsqueeze(0).unsqueeze(-2)

        a_l1 = [nn.Conv2d(1,64,[7,7],stride=(3,2),padding=(3,3)), nn.ReLU(), nn.BatchNorm2d(64), nn.MaxPool2d([4,4],[4,4])]
        a_l2 = [nn.Conv2d(64,256,[3,3],stride=(1,1),padding=(1,1)), nn.ReLU(),
                nn.Conv2d(256,256,[1,1]), nn.ReLU(), nn.BatchNorm2d(256), nn.MaxPool2d([2,2],[2,2])]
        a_l3 = [nn.Conv2d(256,512,[3,3],stride=(1,1),padding=(1,1)), nn.ReLU(),
                nn.Conv2d(512,512,[1,1]), nn.ReLU(), nn.BatchNorm2d(512), nn.AvgPool2d([4,3])]
        self.a_extractor = nn.Sequential(*(a_l1+a_l2+a_l3))

        self.v_align = nn.Sequential(nn.Conv1d(v_dim, hidden_dim, 3, padding=1), nn.ReLU())
        self.i_align = nn.Sequential(nn.Conv1d(f_dim, hidden_dim, 3, padding=1), nn.ReLU())
        self.align_a = nn.Linear(512, hidden_dim, bias=False)
        self.proj_to_hidden = nn.Linear(3*hidden_dim, 4096)

    def forward(self, v_fea, a_fea, f_fea):
        device = v_fea.device
        B, T, H, W = a_fea.shape
        a_fes = []

        # 确保所有输入都在正确的设备上
        v_fea = v_fea.to(device)
        a_fea = a_fea.to(device)
        f_fea = f_fea.to(device)
        tpe = self.tpe.to(device)

        for t in range(0, T, 600):
            tlen = min(600, T-t)
            a_fe = a_fea[:, t:t+tlen] + tpe[:, t:t+tlen]
            
            # 优化内存使用，避免创建不必要的中间张量
            a_roll_1 = a_fe.roll(1, 1)
            a_roll_2 = a_fe.roll(-1, 1)
            
            a_input = torch.cat([
                a_roll_1.view(B*tlen, 1, H, W),
                a_fe.view(B*tlen, 1, H, W),
                a_roll_2.view(B*tlen, 1, H, W)
            ], dim=2)
            
            a_out = self.a_extractor(a_input)
            a_out = torch.flatten(a_out, start_dim=1).contiguous().view(B, tlen, 512)
            a_fes.append(a_out)

            # 清理中间变量
            del a_roll_1, a_roll_2, a_input

        a_fea = torch.cat(a_fes, dim=1)
        del a_fes
        torch.cuda.empty_cache()

        v_fea = self.v_align(v_fea.permute(0, 2, 1)).permute(0, 2, 1)
        f_fea = self.i_align(f_fea.permute(0, 2, 1)).permute(0, 2, 1)
        a_fea_aligned = self.align_a(a_fea)
        
        x = torch.cat([v_fea, a_fea_aligned, f_fea], dim=-1)
        x = self.proj_to_hidden(x)
        
        # 清理不需要的变量
        del v_fea, a_fea, f_fea, a_fea_aligned
        torch.cuda.empty_cache()
        
        return x

class MultiModalClassifier(nn.Module):
    def __init__(self, hidden_dim=4096, num_classes=2):
        super().__init__()
        self.classifier = nn.Linear(hidden_dim, num_classes+1)

    def forward(self, x):
        return self.classifier(x)
def train(train_loader, model, tokenizer, align_module, lora_config,
          num_epochs=5, lr=1e-5, max_grad_norm=1.0, save_dir='./checkpoints'):

    device = torch.device("cuda")
    scaler = GradScaler(device=device)
    classifier = MultiModalClassifier(hidden_dim=4096, num_classes=2).to(device)
    # criterion = nn.BCEWithLogitsLoss()
    # use TemporalEmotionLoss instead of nn.BCEWithLogitsLoss()
    criterion = TemporalEmotionLoss().to(device)

    os.makedirs(save_dir, exist_ok=True)
    model.train()
    align_module.to(device)

    peft_model = PeftModel(model, lora_config).to(device)
    peft_model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        list(peft_model.parameters()) + list(classifier.parameters()),
        lr=lr
    )

    for epoch in range(num_epochs):
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in loop:
            optimizer.zero_grad()
            point_label = batch['point_label'].to(device)
            label=batch['label'].to(device)
            rgb = batch['rgb'].to(device)
            mfcc = batch['mfcc'].to(device)
            img = batch['img'].to(device)

            x_aligned = align_module(rgb, mfcc, img)   # [B, T, hidden]
            B, T, hidden_size = x_aligned.shape
            attention_mask = torch.ones(B, T, device=device)
            # 检查point_label是否全为0
            print(f"No point anno in batch: {[i for i in range(B) if point_label[i].sum() == 0]}")

            with autocast(device_type="cuda"):
                outputs = peft_model(
                    inputs_embeds=x_aligned,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]   # [B, T, hidden_dim]
                logits = classifier(hidden_states)         # [B, T, C]
                # print(logits.shape,"-------------")
                loss = criterion(logits, label, point_label)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(list(peft_model.parameters()) + list(classifier.parameters()), max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            loop.set_postfix(loss=loss.item())

        epoch_save_dir = os.path.join(save_dir, f"epoch_{epoch+1}_lora")
        os.makedirs(epoch_save_dir, exist_ok=True)

        peft_model.save_pretrained(epoch_save_dir)

        tokenizer.save_pretrained(epoch_save_dir)

        torch.save({
            "classifier_state_dict": classifier.state_dict(),
            "align_state_dict": align_module.state_dict()
        }, os.path.join(epoch_save_dir, "classifier_align.pth"))

        print(f"Saved LoRA weights + classifier_align to {epoch_save_dir}")

def load_for_inference(base_model_path, epoch_lora_dir, device="cuda"):
    """
    base_model_path: 基础 Llama 权重目录（同训练时）
    epoch_lora_dir: e.g. '/.../checkpoints/epoch_10_lora'
    返回：peft_model, tokenizer, classifier, align_module
    """
    # 1) 加载 base model & tokenizer（和训练一致）
    tokenizer = LlamaTokenizer.from_pretrained(base_model_path, use_fast=False)
    tokenizer.pad_token = "$$"
    base_model = LlamaForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    base_model = prepare_model_for_int8_training(base_model)

    # 2) 用 PeftModel.from_pretrained 加载 LoRA adapter（会读取 adapter_config.json & adapter_model.bin）
    peft_model = PeftModel.from_pretrained(base_model, epoch_lora_dir)
    peft_model.to(device)
    peft_model.eval()

    # 3) 加载 classifier 和 align_module
    checkpoint = torch.load(os.path.join(epoch_lora_dir, "classifier_align.pth"), map_location=device)
    classifier = MultiModalClassifier(hidden_dim=4096, num_classes=2).to(device)
    align_module = MultiModalAlign().to(device)
    classifier.load_state_dict(checkpoint["classifier_state_dict"])
    align_module.load_state_dict(checkpoint["align_state_dict"])
    classifier.eval()
    align_module.eval()

    return peft_model, tokenizer, classifier, align_module

if __name__=="__main__":
    device = torch.device("cuda")
    feature_paths = [
        './dataset/VideoSenti/features/train/rgb',
        './dataset/VideoSenti/features/train/logmfcc',
        './dataset/VideoSenti/features/train/img'
    ]
    point_csv = './dataset/VideoSenti/point_gaussian/point_labels.csv'

    dataset = EmotionDatasetPoint(feature_paths, point_csv)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    model_path = "./checkpoints/Llama-2-7b" #LLaMA base model path
    model, tokenizer = load_llama_lora(model_path, device=device)
    align_module = MultiModalAlign().to(device)

    # ---------------- LoRA configuration ----------------
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj","v_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
    )

    train(train_loader, model, tokenizer, align_module, lora_config, num_epochs=1000, lr=1e-5)
