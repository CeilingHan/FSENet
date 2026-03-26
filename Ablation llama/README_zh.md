Table 3
## 目录结构

```
Ablation llama/
├── eval/              # 评估相关代码
│   ├── eval_classification.py
│   ├── eval_detection.py
│   └── utils_eval.py
├── dataset/           # 数据集目录（需自行创建）
│   └── VideoSenti/     # VideoSenti数据集
├── checkpoints/       # 模型检查点目录（需自行创建）
│   ├── Llama-2-7b/     # LLaMA-2-7b基础模型
│   └── epoch_*/lora/   # 训练后的LoRA权重
├── config.py          # 配置文件
├── loss.py            # 损失函数
├── main_eval.py       # 主评估脚本
├── test.py            # 测试脚本
├── test_all.py        # 批量测试脚本
├── tsl.py             # TSL模型实现
├── tsl_llama.py       # TSL-LLaMA模型实现
├── tsl_loss.py        # TSL损失函数
├── utils.py           # 工具函数
└── zero-shot.py       # 零样本学习脚本
```

## 环境要求

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- NumPy
- Pandas
- tqdm

## 数据集准备

本项目使用VideoSenti数据集，需要按照以下结构组织：

```
dataset/VideoSenti/
├── features/           # 特征目录
│   ├── train/          # 训练集特征
│   │   ├── rgb/        # RGB特征
│   │   ├── logmfcc/    # MFCC特征
│   │   └── img/        # 人脸特征
│   └── test/           # 测试集特征
│       ├── rgb/        # RGB特征
│       ├── logmfcc/    # MFCC特征
│       └── img/        # 人脸特征
├── point_gaussian/     # 点标注目录
│   ├── point_labels.csv      # 训练集点标注
│   └── test_labels.csv       # 测试集点标注
├── gt.json             # 测试集 ground truth
└── fps_dict.json       # 视频帧率信息
```

## 模型训练

### 1. 准备基础模型

下载LLaMA-2-7b模型，并放置在`checkpoints/Llama-2-7b/`目录下。

#### LLaMA-2-7b 模型下载

LLaMA-2-7b有英文和中文版本可供选择：

**英文版本**：
- 从Hugging Face下载：[meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- 从Meta官方网站申请：[Meta AI](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)


**注意**：下载后请将模型文件放置在`checkpoints/Llama-2-7b/`目录中，确保目录结构正确。

### 2. 训练脚本

使用`tsl_llama.py`进行模型训练：

```bash
python tsl_llama.py
```

训练过程中，模型会保存在`checkpoints/epoch_*/lora/`目录下，每个epoch保存一次。

## 模型评估

### 1. 单模型评估

使用`test.py`评估单个模型：

```bash
python test.py
```

### 2. 批量模型评估

使用`test_all.py`评估多个模型：

```bash
python test_all.py
```

## 零样本学习

使用`zero-shot.py`进行零样本情感识别：

```bash
python zero-shot.py
```

**注意**：需要修改`zero-shot.py`中的路径配置：
- `model_path`：LLaMA-2-7b模型路径
- `output_json_path`：输出结果路径
- `feature_paths`：特征路径列表
