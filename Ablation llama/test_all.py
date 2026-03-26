import os
import torch
import numpy as np
import json
import sys
from torch.utils.data import DataLoader
from eval.eval_detection import ANETdetection
from tsl_llama import EmotionDatasetPoint, MultiModalAlign, MultiModalClassifier
from transformers import LlamaTokenizer, LlamaForCausalLM
from peft import PeftModel
import utils


def load_trained_model(model_path, lora_path, checkpoint_path, device='cuda'):
    """
    load trained model (LLaMA + LoRA + align module + classifier)
    """
    print("=== Loading base LLaMA model ===")
    tokenizer = LlamaTokenizer.from_pretrained(model_path, use_fast=False)
    tokenizer.pad_token = "$$"

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32,
        device_map="auto"
    )
   

    print("=== Loading LoRA adapter ===")
    model = PeftModel.from_pretrained(model, lora_path)
    model = model.float()



    print("=== Loading classifier and align module ====")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 检查checkpoint是否包含必要的键
    if 'align_state_dict' not in checkpoint:
        raise KeyError("checkpoint missing 'align_state_dict' key")
    if 'classifier_state_dict' not in checkpoint:
        raise KeyError("checkpoint missing 'classifier_state_dict' key")

    align_module = MultiModalAlign().to(device)
    align_module.load_state_dict(checkpoint['align_state_dict'])
    align_module = align_module.float()

    classifier = MultiModalClassifier(hidden_dim=4096, num_classes=2).to(device)
    classifier.load_state_dict(checkpoint['classifier_state_dict'])
    classifier = classifier.float()

    return model, align_module, classifier

def evaluate_lora(model, align_module, classifier, feature_paths, config, output_path="./results"):
    device = torch.device("cuda")
    model.eval()
    align_module.eval()
    classifier.eval()

    os.makedirs(output_path, exist_ok=True)
    log_file = os.path.join(output_path, "evaluation_log.txt")

    # ---- log redirect ----
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
        def flush(self):
            for f in self.files:
                f.flush()

    log_f = open(log_file, 'w')
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_f)

    try:
        # ---- data load ----
        dataset = EmotionDatasetPoint(feature_paths, point_csv=config.point_csv, mode='test')
        test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        final_res = {
            "version": "VERSION 1.3",
            "results": {},
            "external_data": {"used": True, "details": "VideoSenti Features"}
        }

        fps_dict = json.load(open(os.path.join(config.data_path, 'fps_dict.json')))

        with torch.no_grad():
            for batch in test_loader:
                vid_name = batch['vid'][0]
                rgb = batch['rgb'].to(device)
                mfcc = batch['mfcc'].to(device)
                img = batch['img'].to(device)

                # ---- modal align ----
                x_aligned = align_module(rgb, mfcc, img)
                B, T, H = x_aligned.shape
                attention_mask = torch.ones(B, T, device=device)

                # ---- llama feature extract ----
                outputs = model(
                    inputs_embeds=x_aligned,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                hidden_states = outputs.hidden_states[-1]  # [B, T, H]

                # ---- classifier ----
                logits = classifier(hidden_states).float()  # [B, T, C+1]
                scores = torch.sigmoid(logits).mean(dim=1)  # [B, C+1]
                scores=scores[:,:-1]
                scores_np = scores[0].cpu().numpy()

                # ---- classification prediction ----
                pred = np.where(scores_np >= config.class_thresh)[0]
                if len(pred) == 0:
                    pred = np.array([np.argmax(scores_np)])

                # ---- 2. CAS time sequence ----
                cas_np = torch.sigmoid(logits)[0].cpu().numpy()  # [T, C+1]
                cas_pred = np.expand_dims(cas_np[:, pred], axis=-1)  # only take predicted class
                cas_pred = utils.upgrade_resolution(cas_pred, config.scale)

                # ---- 3. Agnostic score ----
                # background dimension at last
                agnostic_score = 1 - torch.sigmoid(logits)[:, :, -1:]  # [B, T, 1]
                agnostic_score = agnostic_score.expand(-1, -1, config.num_classes)  # [B, T, C]
                agnostic_np = np.expand_dims(agnostic_score[0].cpu().numpy()[:, pred], axis=-1)
                agnostic_np = utils.upgrade_resolution(agnostic_np, config.scale)


                # ---- Proposal dict ----
                proposal_dict = {}

                # CAS threshold proposals
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

                # Agnostic threshold proposals
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

                # ---- NMS ----
                final_proposals = []
                for class_id, class_proposals in proposal_dict.items():
                    # flatten internal possible multiple proposal list
                    flat_proposals = []
                    for p in class_proposals:
                        if isinstance(p[0], (list, tuple, np.ndarray)):
                            flat_proposals.extend(p)
                        else:
                            flat_proposals.append(p)

                    if len(flat_proposals) == 0:
                        continue

                    proposal_array = np.array(flat_proposals)  # shape [N, 4]
                    nmsed = utils.nms(proposal_array, thresh=0.5)
                    final_proposals.extend(nmsed)

                print(f"{vid_name} proposals saved: {len(final_proposals)}")
                final_res["results"][vid_name] = utils.result2json(final_proposals)

        # ---- save JSON ----
        json_path = os.path.join(output_path, "temp_result.json")
        with open(json_path, 'w') as f:
            json.dump(final_res, f)

        # ---- 计算 mAP/F2 ----
        tIoU_thresh = np.linspace(0.1, 0.3, 5)
        anet_eval = ANETdetection(config.gt_path,
                                   json_path,
                                   subset='test', tiou_thresholds=tIoU_thresh,
                                   verbose=False, check_status=False)
        mAP, _, info = anet_eval.evaluate()
        cAP, _, Rc, F2 = info

        # avoid F2 nan
        F2 = np.nan_to_num(F2)
        Rc = np.nan_to_num(Rc)

        print(f"Average mAP[0.1:0.3]: {mAP.mean():.4f}, Recall: {Rc.mean():.4f}, F2: {F2.mean():.4f}")

        return final_res, mAP, Rc, F2

    finally:
        sys.stdout = original_stdout
        log_f.close()

if __name__ == "__main__":
    feature_paths = [
        os.path.join('./dataset/VideoSenti', 'features', 'test', 'rgb'),
        os.path.join('./dataset/VideoSenti', 'features', 'test', 'logmfcc'),
        os.path.join('./dataset/VideoSenti', 'features', 'test', 'img')
    ]

    class Config:
        data_path = './dataset/VideoSenti'#VideoSenti dataset path
        point_csv = None#point csv file path
        gt_path = os.path.join('./dataset/VideoSenti', 'gt.json')#ground truth file path
        output_path = './results'#output path
        class_thresh = 0.5#class threshold
        scale = 24#scale factor
        act_thresh_cas = np.arange(0, 0.25, 0.025)#CAS threshold
        act_thresh_agnostic = np.arange(0.4, 0.75, 0.025)#agnostic threshold
        num_classes = 2#number of classes

    config = Config()
    # === model path ===
    model_path = "./checkpoints/Llama-2-7b"#LLaMA base model path
    lora_root = "./checkpoints"  # lora directory root
    checkpoint_template = "epoch_{}_lora/classifier_align.pth"

    best_mAP = -1.0
    best_epoch = None
    all_results = []
    best_result= None

    # 扩展epoch_id列表以评估更多模型
    for epoch_id in [10, 15, 20, 25, 30]:  # 根据实际训练的epoch范围调整
        checkpoint_path = os.path.join(lora_root, f"epoch_{epoch_id}_lora", "classifier_align.pth")
        if not os.path.exists(checkpoint_path):
            print(f"[WARNING] {checkpoint_path} does not exist, skip.")
            continue
        torch.cuda.empty_cache()
        print(f"Evaluating checkpoint: epoch_{epoch_id}_lora")
        model, align_module, classifier = load_trained_model(model_path, os.path.join(lora_root, f"epoch_{epoch_id}_lora"), checkpoint_path)



        final_res, mAP, Rc, F2 = evaluate_lora(model, align_module, classifier, feature_paths, config)
        print("=================",mAP, Rc, F2,"================")
        mean_mAP = mAP.mean()
        all_results.append((epoch_id, mean_mAP, Rc.mean(), F2.mean()))
        
        # 更新最优
        if mean_mAP > best_mAP:
            best_mAP = mean_mAP
            best_epoch = epoch_id
            best_result = (epoch_id, mean_mAP, Rc.mean(), F2.mean())

        # 保存每次的日志
        log_txt = os.path.join(config.output_path, "evaluation_our_epochs.txt")
        with open(log_txt, 'a') as f:
            f.write(f"epoch_{epoch_id}_lora: Average mAP[0.1:0.3]: {mean_mAP:.7f}, Recall: {Rc.mean():.7f}, F2: {F2.mean():.7f}\n")
    
        del model, align_module, classifier
        torch.cuda.empty_cache()
