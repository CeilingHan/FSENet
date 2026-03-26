import os
import torch
import numpy as np
import json
from torch.utils.data import DataLoader
from eval.eval_detection import ANETdetection
from tsl_llama import EmotionDatasetPoint, MultiModalAlign
import utils

def load_model_fixed(model_path, lora_path="./checkpoints/epoch_100_lora"):
    from transformers import LlamaForCausalLM
    from peft import PeftModel
    print("Loading base model...")
    base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,  # 强制使用float32
        device_map="auto"
    )
    
    print("Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, lora_path)
    
    model = model.float()
    
    if hasattr(model, 'lm_head'):
        if hasattr(model.lm_head, 'module'):
            print("Replacing wrapped lm_head with base module...")
            model.lm_head = model.lm_head.module
        
        model.lm_head = model.lm_head.float()
    
    # 加载align_module权重
    checkpoint_path = os.path.join(lora_path, "classifier_align.pth")
    if os.path.exists(checkpoint_path):
        print("Loading align module weights...")
        checkpoint = torch.load(checkpoint_path, map_location='cuda')
        align_module = MultiModalAlign().to('cuda').float()
        align_module.load_state_dict(checkpoint['align_state_dict'])
    else:
        print("Warning: classifier_align.pth not found, using fresh align module")
        align_module = MultiModalAlign().to('cuda').float()
    
    return model, align_module

def evaluate_lora(model, align_module, feature_paths, config, output_path="./results"):
    device = torch.device("cuda")
    model.eval()
    align_module.eval()
    
    os.makedirs(output_path, exist_ok=True)
    
    dataset = EmotionDatasetPoint(feature_paths, point_csv=config.point_csv, mode='test')
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    final_res = {"version": "VERSION 1.3", "results": {}, "external_data": {"used": True, "details": "VideoSenti Features"}}
    fps_dict = json.load(open(os.path.join(config.data_path, 'fps_dict.json')))
    
    with torch.no_grad():
        for batch in test_loader:
            vid_name = batch['vid'][0]
            rgb = batch['rgb'].to(device)
            mfcc = batch['mfcc'].to(device)
            img = batch['img'].to(device)
            
            x_aligned = align_module(rgb, mfcc, img)
            
            B, T, H = x_aligned.shape
            attention_mask = torch.ones(B, T, device=device)
            
            outputs = model(inputs_embeds=x_aligned, attention_mask=attention_mask)
            logits = outputs.logits[:, :T, :config.num_classes].float()
            scores = torch.sigmoid(logits).mean(dim=1)
            
            scores_np = scores[0].cpu().numpy()
            
            pred = np.where(scores_np >= config.class_thresh)[0]
            if len(pred) == 0:
                pred = np.array([np.argmax(scores_np)])
            
            cas_np = torch.sigmoid(logits)[0].cpu().numpy()
            cas_pred = np.expand_dims(cas_np[:, pred], axis=-1)
            cas_pred = utils.upgrade_resolution(cas_pred, config.scale)
            
            agnostic_score = 1 - torch.sigmoid(logits)[:, :, -1:].float()
            agnostic_score = agnostic_score.expand((-1, -1, config.num_classes))
            agnostic_np = np.expand_dims(agnostic_score[0].cpu().numpy()[:, pred], axis=-1)
            agnostic_np = utils.upgrade_resolution(agnostic_np, config.scale)
            
            # Proposal dict
            proposal_dict = {}
            
            # CAS threshold
            for thr in config.act_thresh_cas:
                cas_temp = cas_pred.copy()
                cas_temp[cas_temp[:, :, 0] < thr] = 0
                seg_list = [np.where(cas_temp[:, c, 0] > 0) for c in range(len(pred))]
                proposals = utils.get_proposal_oic(seg_list, cas_temp, scores_np, pred,
                                                   config.scale, T, fps_dict[vid_name], T)
                for p in proposals:
                    class_id = int(p[0]) if not isinstance(p[0], (list, np.ndarray)) else int(p[0][0])
                    if class_id not in proposal_dict:
                        proposal_dict[class_id] = []
                    proposal_dict[class_id].append(p)
            
            # Agnostic threshold
            for thr in config.act_thresh_agnostic:
                agnostic_temp = agnostic_np.copy()
                agnostic_temp[agnostic_temp[:, :, 0] < thr] = 0
                seg_list = [np.where(agnostic_temp[:, c, 0] > 0) for c in range(len(pred))]
                proposals = utils.get_proposal_oic(seg_list, cas_pred, scores_np, pred,
                                                   config.scale, T, fps_dict[vid_name], T)
                for p in proposals:
                    class_id = int(p[0]) if not isinstance(p[0], (list, np.ndarray)) else int(p[0][0])
                    if class_id not in proposal_dict:
                        proposal_dict[class_id] = []
                    proposal_dict[class_id].append(p)
            
            


            # NMS
            final_proposals = []
            for class_id, class_proposals in proposal_dict.items():
    
                print("---",class_proposals)
                print(f"Class {class_id} proposals: {len(class_proposals)}")
                
                if len(class_proposals) > 0:
                    proposal_array = np.array(class_proposals)
                    
                    if proposal_array.ndim == 3:
                        proposal_array = proposal_array.squeeze(1)
                    
                    print(f"Proposal array shape: {proposal_array.shape}")
                    
                    nmsed = utils.nms(proposal_array, thresh=0.5)
                    final_proposals.extend(nmsed)
            
            final_res["results"][vid_name] = utils.result2json(final_proposals)
    
    json_path = os.path.join(output_path, "temp_result.json")
    with open(json_path, 'w') as f:
        json.dump(final_res, f)
    
    # ANETdetection evaluation
    tIoU_thresh = np.linspace(0.1, 0.3, 5)
    anet_eval = ANETdetection(config.gt_path, json_path,
                               subset='test', tiou_thresholds=tIoU_thresh,
                               verbose=False, check_status=False)
    mAP, _, info = anet_eval.evaluate()
    cAP, _, Rc, F2 = info
    
    print(f"Average mAP[0.1:0.3]: {mAP.mean():.4f}, Recall: {Rc.mean():.4f}, F2: {F2.mean():.4f}")
    return final_res, mAP, Rc, F2

# ================= Config & Run =================
if __name__=="__main__":
    feature_paths = [
        './dataset/VideoSenti/features/test/rgb',
        './dataset/VideoSenti/features/test/logmfcc',
        './dataset/VideoSenti/features/test/img'
    ]

    class Config:
        data_path = './dataset/VideoSenti' #VideoSenti dataset path
        point_csv = './dataset/VideoSenti/point_gaussian/test_labels.csv' #test_labels csv file path
        gt_path = './dataset/VideoSenti/gt.json' #ground truth file path
        class_thresh = 0.5
        scale = 24
        act_thresh_cas = np.arange(0, 0.25, 0.025)
        act_thresh_agnostic = np.arange(0.4, 0.75, 0.025)
        output_path = './results'
        num_classes = 2

    config = Config()

    model_path = "./checkpoints/Llama-2-7b" #LLaMA base model path
    model, align_module = load_model_fixed(model_path)

    final_res, mAP, Rc, F2 = evaluate_lora(model, align_module, feature_paths, config)