Only the Table 3
## Directory Structure

```
Ablation llama/
├── eval/              # Evaluation related code
│   ├── eval_classification.py
│   ├── eval_detection.py
│   └── utils_eval.py
├── dataset/           # Dataset directory (need to create)
│   └── VideoSenti/     # VideoSenti dataset
├── checkpoints/       # Model checkpoints directory (need to create)
│   ├── Llama-2-7b/     # LLaMA-2-7b base model
│   └── epoch_*/lora/   # Trained LoRA weights
├── config.py          # Configuration file
├── loss.py            # Loss function
├── main_eval.py       # Main evaluation script
├── test.py            # Test script
├── test_all.py        # Batch test script
├── tsl.py             # TSL model implementation
├── tsl_llama.py       # TSL-LLaMA model implementation
├── tsl_loss.py        # TSL loss function
├── utils.py           # Utility functions
└── zero-shot.py       # Zero-shot learning script
```

## Environment Requirements

- Python 3.9+
- PyTorch 2.0+
- Transformers 4.30+
- PEFT 0.4+
- NumPy
- Pandas
- tqdm

## Dataset Preparation

This project uses the VideoSenti dataset, which needs to be organized according to the following structure:

```
dataset/VideoSenti/
├── features/           # Features directory
│   ├── train/          # Training set features
│   │   ├── rgb/        # RGB features
│   │   ├── logmfcc/    # MFCC features
│   │   └── img/        # Face features
│   └── test/           # Test set features
│       ├── rgb/        # RGB features
│       ├── logmfcc/    # MFCC features
│       └── img/        # Face features
├── point_gaussian/     # Point annotation directory
│   ├── point_labels.csv      # Training set point annotations
│   └── test_labels.csv       # Test set point annotations
├── gt.json             # Test set ground truth
└── fps_dict.json       # Video frame rate information
```

## Model Training

### 1. Prepare Base Model

Download the LLaMA-2-7b model and place it in the `checkpoints/Llama-2-7b/` directory.

#### LLaMA-2-7b Model Download

LLaMA-2-7b is available in both English and Chinese versions:

**English Version**:
- Download from Hugging Face: [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf)
- Apply from Meta official website: [Meta AI](https://ai.meta.com/resources/models-and-libraries/llama-downloads/)

### 2. Training Script

Use `tsl_llama.py` for model training:

```bash
python tsl_llama.py
```

During training, models will be saved in the `checkpoints/epoch_*/lora/` directory, with one save per epoch.

## Model Evaluation

### 1. Single Model Evaluation

Use `test.py` to evaluate a single model:

```bash
python test.py
```

### 2. Batch Model Evaluation

Use `test_all.py` to evaluate multiple models:

```bash
python test_all.py
```

## Zero-shot Learning

Use `zero-shot.py` for zero-shot emotion recognition:

```bash
python zero-shot.py
```

**Note**: Need to modify path configurations in `zero-shot.py`:
- `model_path`: LLaMA-2-7b model path
- `output_json_path`: Output result path
- `feature_paths`: Feature path list

## Ablation Experiments

### Experimental Setup

This project conducted the following ablation experiments:

1. **Different model architectures**: Comparing Qwen-3.0, LLaMA-2-7B, and our model
2. **Different training methods**: Comparing zero-shot learning and LoRA fine-tuning
3. **Different feature combinations**: Evaluating the combination effects of RGB, MFCC, and face features
