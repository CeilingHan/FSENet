import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import LlamaForCausalLM, LlamaTokenizer, AdamW
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training
import numpy as np
 
class EmotionDataset(torch.utils.data.Dataset):
    def __init__(self, data_json, data_path, modal='all', sampling='uniform', num_sample_clips=8):
        """
        data_json: JSON dict, each video has multiple annotations
        modal: 'all' or 'rgb'/'logmfcc'/'img'
        """
        self.data = data_json["database"]
        self.modal = modal
        self.sampling = sampling
        self.num_sample_clips = num_sample_clips
        self.label_map = {'p':1, 'n':0}  # positive=1, negative=0

        if self.modal == 'all':
            self.feature_path = []
            for _modal in ['rgb', 'logmfcc', 'img']:
                feature_dir = os.path.join(data_path, 'features', _modal)
                self.feature_path.append(feature_dir)
        else:
            self.feature_path = os.path.join(data_path, 'features', self.modal)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        vid_name = list(self.data.keys())[index]
        info = self.data[vid_name]
        annots = info.get("annotations", [])

        # load features
        rgb = np.load(os.path.join(self.feature_path[0], vid_name + '.npy')).astype(np.float32)
        audio = np.load(os.path.join(self.feature_path[1], vid_name + '.npy')).astype(np.float32)
        image = np.load(os.path.join(self.feature_path[2], vid_name + '.npy')).astype(np.float32)

        T = min(rgb.shape[0], int(audio.shape[0]/32), image.shape[0])
        rgb = rgb[:T]
        image = image[:T]
        audio = audio[:T*32].reshape(T,32,60)
        audio = (audio + 50)/80

        idxs = self._sample_indices(T)
        rgb = rgb[idxs]
        audio = audio[idxs]
        image = image[idxs]

        label, point_label, label_distribution = self.get_label(index, T, idxs)
        # label, point_label, label_distribution = self.get_label(index, T, idxs)

        features = {"rgb": torch.tensor(rgb), "audio": torch.tensor(audio), "image": torch.tensor(image)}
        return {"vid": vid_name, "features": features, "label": label, "point_label": point_label, "label_distribution": label_distribution}
    def _sample_indices(self, length):
        if self.sampling == 'random':
            idx = np.linspace(0, length-1, self.num_sample_clips)
            perturb = np.random.randint(-1,2, size=idx.shape)
            idx = np.clip(idx + perturb, 0, length-1).astype(int)
        else:
            idx = np.linspace(0, length-1, self.num_sample_clips).astype(int)
        return idx

    def get_label(self, index, vid_num_seg, sample_idx):
        vid_name = self.vid_list[index]

        # hard label for video
        label = np.zeros([self.num_classes], dtype=np.float32)
        label_distribution = np.zeros([self.num_classes], dtype=np.float32)

        # supervision: point-level
        temp_anno = np.zeros([vid_num_seg, self.num_classes], dtype=np.float32)
        t_factor = 1/16  # 映射 frame -> sampled idx

        temp_df = self.point_anno[self.point_anno["video_id"]==vid_name][['point','class_index']]

        for key in temp_df.index:
            point = temp_df.at[key,'point']
            class_idx = temp_df.at[key,'class_index']
            idx = int(point * t_factor)
            if idx >= vid_num_seg:
                idx = vid_num_seg-1
            temp_anno[idx,:] = 0
            temp_anno[idx,class_idx] = 1
            label[class_idx] = 1
            label_distribution[class_idx] += 1

        label_distribution = label_distribution / max(label_distribution.sum(),1e-6)
        point_label = temp_anno[sample_idx,:]

        return torch.tensor(label), torch.from_numpy(point_label), torch.tensor(label_distribution)


class EmotionLLaMA(nn.Module):
    def __init__(self, llama_model_path, hidden_size=4096, lora_r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.llama = LlamaForCausalLM.from_pretrained(
            llama_model_path,
            torch_dtype=torch.float16
        )
        self.tokenizer = LlamaTokenizer.from_pretrained(llama_model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        if lora_r > 0:
            self.llama = prepare_model_for_int8_training(self.llama)
            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=["q_proj","v_proj"],
                task_type="CAUSAL_LM",
                lora_dropout=lora_dropout
            )
            self.llama = get_peft_model(self.llama, lora_config)

        # feature projection
        self.rgb_proj = nn.Linear(512, hidden_size)
        self.audio_proj = nn.Linear(32*60, hidden_size)
        self.image_proj = nn.Linear(256, hidden_size)
        self.cls_proj = nn.Linear(hidden_size*3, hidden_size)

        # classification head
        self.classifier = nn.Linear(hidden_size, 2)  # binary classification

    def forward(self, features):
        rgb = features['rgb'].half()      # [B,T,D1]
        audio = features['audio'].half()  # [B,T,32,60]
        image = features['image'].half()  # [B,T,D3]

        B, T, _ = rgb.shape
        rgb_emb = self.rgb_proj(rgb)                     # [B,T,H]
        audio_emb = self.audio_proj(audio.view(B,T,-1)) # [B,T,H]
        image_emb = self.image_proj(image)              # [B,T,H]

        combined = torch.cat([rgb_emb, audio_emb, image_emb], dim=-1)   # [B,T,3*H]
        cls_token = self.cls_proj(combined[:,0,:]).unsqueeze(1)         # [B,1,H]
        emb = torch.cat([combined, cls_token], dim=1)                    # [B,T+1,H]

        logits = self.classifier(emb)  # [B,T+1,2]

        return logits[:, :-1, :]



def train():
    data_path = "./dataset/Emotion" #Emotion dataset path
    train_json = os.path.join(data_path, "train_data.json")
    with open(train_json,"r") as f:
        data_json = json.load(f)

    dataset = EmotionDataset(data_json, data_path, modal='all', sampling='uniform')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=5e-5)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):
        for batch_idx, batch in enumerate(dataloader):
            features = {k: batch['features'][k].to(device) for k in batch['features']}
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast():  # 半精度训练
                logits = model(features)
                loss = criterion(logits.view(-1,2), labels.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch} Batch {batch_idx} Loss {loss.item():.4f}")

        torch.save(model.state_dict(), f"checkpoint_epoch{epoch}.pth")
        print(f"Saved checkpoint_epoch{epoch}.pth")


if __name__ == "__main__":
    llama_path = "./checkpoints/Llama-2-7b" #LLaMA base model path
    model = EmotionLLaMA(llama_path, hidden_size=4096, lora_r=8)
    train()
