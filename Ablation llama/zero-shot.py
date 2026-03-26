import os
import json
import numpy as np
import torch
from tqdm import tqdm
from transformers import LlamaForCausalLM, LlamaTokenizer
from torch.utils.data import DataLoader
import re

# ================= 配置 =================
# 请根据实际情况修改以下路径
model_path = "./checkpoints/Llama-2-7b" # Llama-2-7b模型本地路径
output_json_path = "./results/zero_shot_results.json" # 输出结果路径

# 特征路径列表
feature_paths = [
    './dataset/VideoSenti/features/test/rgb',      # RGB特征路径
    './dataset/VideoSenti/features/test/logmfcc',   # MFCC特征路径
    './dataset/VideoSenti/features/test/img'       # 人脸特征路径
]

from tsl_llama import EmotionDatasetPoint
dataset = EmotionDatasetPoint(feature_paths, mode='test')
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="auto")
model.eval()


def describe_mfcc_segment(mfcc_segment):
    """
    describe the MFCC segment
    """
    mfcc_segment = mfcc_segment.reshape(mfcc_segment.shape[0], -1)
    energy = np.linalg.norm(mfcc_segment, axis=1)
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    trend = np.polyfit(np.arange(len(energy)), energy, 1)[0]
    
    if trend > 0.01:
        trend_desc = "increasing energy over time"
    elif trend < -0.01:
        trend_desc = "decreasing energy over time"
    else:
        trend_desc = "stable energy"

    variability = std_energy / (mean_energy + 1e-6)
    if variability < 0.05:
        var_desc = "very stable"
    elif variability < 0.15:
        var_desc = "moderately stable"
    else:
        var_desc = "highly fluctuating"

    return f"MFCC audio features: {trend_desc}, {var_desc}, mean energy={mean_energy:.3f}"

def describe_feature(feat, name):
    feat = np.array(feat)
    T = feat.shape[0]

    energy = np.linalg.norm(feat, axis=1)
    energy_mean = np.mean(energy)
    std_energy = np.std(energy)

    # safer trend estimation
    if T < 2 or np.allclose(energy, energy[0]):
        trend = 0.0
    else:
        try:
            trend = (energy[-1] - energy[0]) / (T-1)
        except:
            trend = 0.0

    variability = std_energy / (energy_mean + 1e-6)

    if trend > 0.01:
        trend_desc = "increasing energy"
    elif trend < -0.01:
        trend_desc = "decreasing energy"
    else:
        trend_desc = "stable energy"

    if variability < 0.05:
        var_desc = "very stable"
    elif variability < 0.15:
        var_desc = "moderately stable"
    else:
        var_desc = "highly fluctuating"

    # 模态提示
    if name == "rgb":
        sem = "body movement and visual actions"
    elif name == "mfcc":
        sem = "speech tone and acoustic energy"
    elif name == "face":
        sem = "facial expression intensity"
    else:
        sem = "general signals"

    return f"{name}: {trend_desc}, {var_desc}, average energy={energy_mean:.3f}, related to {sem}"
def feature_to_summary_segmented(rgb, mfcc, face, num_segments=8, fps=30):
    T = rgb.shape[0]
    seg_len = T // num_segments
    summaries = []

    for i in range(num_segments):
        start = i * seg_len
        end = (i + 1) * seg_len if i < num_segments - 1 else T
        seg_time = [round(start / fps, 2), round(end / fps, 2)]
        rgb_sum = describe_feature(rgb[start:end], "rgb")
        mfcc_sum = describe_feature(mfcc[start:end].reshape(end - start, -1), "mfcc")
        face_sum = describe_feature(face[start:end], "face")
        summaries.append({
            "segment": seg_time,
            "summary_text": f"Segment {i+1} ({seg_time[0]}s–{seg_time[1]}s):\n"
                            f"- {rgb_sum}\n- {mfcc_sum}\n- {face_sum}"
        })
    return summaries

prompt_template = """You are an expert in temporal emotion recognition.
Analyze the emotional polarity (positive or negative) for each segment of the video, based on multimodal features.

Video ID: {video_id}

Below are the summaries of each segment extracted from the video:
{segment_summaries}

Example of desired output:
[
  {{"label": "positive", "score": 0.85, "segment": [0.0, 5.0]}},
  {{"label": "negative", "score": 0.3, "segment": [5.0, 10.0]}}
]

Now provide the JSON output for the current video.

Only output JSON content, do not include any explanation.
"""

results = {"results": {}}

for batch in tqdm(dataloader):
    vid = batch["vid"][0]
    rgb = batch["rgb"][0].numpy()
    mfcc = batch["mfcc"][0].numpy()
    face = batch["img"][0].numpy()

    seg_summaries = feature_to_summary_segmented(rgb, mfcc, face, num_segments=8)
    seg_text = "\n".join([s["summary_text"] for s in seg_summaries])
    prompt = prompt_template.format(video_id=vid, segment_summaries=seg_text)

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.4, do_sample=True)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("------ MODEL RAW OUTPUT ------")
    print(decoded)
    print("------------------------------")
    match = re.search(r"\[\s*\{.*\}\s*\]", decoded, re.DOTALL)
    if match:
        try:
            json_part = json.loads(match.group())
        except:
            json_part = []
    else:
        json_part = []

    results["results"][vid] = json_part

os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
with open(output_json_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"✅ Saved segment-level results to {output_json_path}")
