# PTCD

This repository contains the official implementation for the paper "Towards Transferable Personality Representation Learning based on Triplet Comparisons and Its Applications" (EMNLP-Main 2025).

## Environment Requirements

The code requires the following dependencies (see `requirements.txt`):

```
torch>=1.8.0
numpy
pandas
scikit-learn
transformers
datasets
tqdm
scipy
```

## Datasets

**Download:**
The datasets are available at Google Drive:
[Download Datasets](https://drive.google.com/drive/folders/1FMXuSmEsxTpKgvuEZYSN8a5ijH0k7hr1?usp=drive_link)

**Organization:**
The dataset contains three main folders:
1. `utterances`: Raw single-sentence corpora.
2. `triplet`: Generated and filtered triplets used for encoder training.
3. `by-product`: By-product datasets used for downstream verification (personality detection).

After downloading, please organize the data files into the `data/` directory.

## Training Pipeline

The training process consists of Pre-training (Contrastive Learning) and Downstream Personality Detection.

### Step 1: Pre-training (Embedding)

We use contrastive learning to fine-tune the BERT embeddings.

**1. Warm-up Training:**
This step performs Masked Language Modeling (MLM) or similar warm-up tasks.

```bash
cd embedding
bash scripts/train_warm_ml.sh
```

**2. Contrastive Pre-training:**
Train the encoder using contrastive loss.

```bash
cd embedding
bash scripts/train.sh
```

The trained model will be saved in `embedding/output_embedding/` (or as configured in the script).

### Step 2: Downstream Personality Detection

Train the classifier for personality traits (e.g., MBTI/Big5) using the pre-trained embeddings.

**1. Train:**
```bash
cd personality_detection
bash scripts/run.sh
```

**2. Test:**
```bash
cd personality_detection
bash scripts/test.sh
```

## Citation

If you find this code useful, please cite our paper:

```bibtex
@inproceedings{tang2025towards,
  title = {Towards Transferable Personality Representation Learning based on Triplet Comparisons and Its Applications},
  author = {Kai Tang and Rui Wang and Renyu Zhu and Minmin Lin and Xiao Ding and Tangjie Lv and Changjie Fan and Runze Wu and Haobo Wang},
  booktitle = {The Conference on Empirical Methods in Natural Language Processing (EMNLP-Main)},
  year = {2025}
}
```

