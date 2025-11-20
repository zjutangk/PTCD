import warnings

warnings.filterwarnings("ignore")
import logging
import random
import os
import sys
import time
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Union
import re
from datasets import load_dataset, Dataset, concatenate_datasets
from torch import nn
import codecs
from model.bert import BertForModel

# from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, classification_report
import transformers
from transformers import (
    Trainer,
    set_seed,
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    AutoModel,
)
from bert import RobertaForTripleTextEncoding, BertForTripleTextEncoding

work_dir = os.path.dirname(os.path.realpath(__file__))
# setup logging
transformers.logging.set_verbosity_info()
transformers.logging.enable_default_handler()
transformers.logging.enable_explicit_format()
transformers.logging.add_handler(logging.StreamHandler(sys.stdout))
logger = transformers.logging.get_logger()


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    # print("logits, labels", logits, labels)
    logits = np.squeeze(logits)
    logits = logits > 0
    acc = sum(logits) / len(logits)
    return {"acc": acc}


@dataclass
class OtherArguments:
    use_mllpretrain: int
    train_file: List[str]
    valid_file: List[str]
    test_file: List[str] = field(default="")
    model_dir: Optional[str] = field(
        default=work_dir + "/model_dir",
    )


@dataclass
class DataCollator:
    tokenizer: AutoTokenizer
    max_input_length: int = 512

    def __call__(self, examples):
        batch = {}
        num = len(examples)
        anchor_texts = []
        positive_texts = []
        negative_texts = []
        labels = []
        for i, example in enumerate(examples):
            anchor = example["背景"]
            text0 = example["选项0"]
            text1 = example["选项1"]
            label = example["label"]
            labels.append(label)
            if label == 0:
                positive = text0
                negative = text1
            elif label == 1:
                positive = text1
                negative = text0
            else:
                print("label is wrong")
            anchor_texts.append(anchor)
            positive_texts.append(positive)
            negative_texts.append(negative)

        anchor_inputs = self.tokenizer(
            anchor_texts,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        positive_inputs = self.tokenizer(
            positive_texts,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        negative_inputs = self.tokenizer(
            negative_texts,
            max_length=self.max_input_length,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
            add_special_tokens=True,
        )
        for k, v in anchor_inputs.items():
            batch[f"anchor_{k}"] = v
        for k, v in positive_inputs.items():
            batch[f"positive_{k}"] = v
        for k, v in negative_inputs.items():
            batch[f"negative_{k}"] = v

        batch["labels"] = torch.tensor(labels, dtype=torch.long)

        return batch


def main():
    parser = HfArgumentParser((TrainingArguments, OtherArguments))
    training_args, other_args = parser.parse_args_into_dataclasses()
    training_args.dataloader_prefetch_factor = None
    # training_args.neftune_noise_alpha = 0.1

    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"other_args {other_args}")

    # Set seed before initializing model.
    set_seed(training_args.seed)
    config = AutoConfig.from_pretrained(other_args.model_dir)
    tokenizer = AutoTokenizer.from_pretrained(other_args.model_dir, use_fast=True)

    print(
        f"-----------------------config.model_type:{config.model_type}----------------------------"
    )

    AutoModel = BertForTripleTextEncoding
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(other_args.model_dir, config=config)
    # get dict from single-sentence pretrain
    if other_args.use_mllpretrain:

        model_path = os.path.join(other_args.model_dir, "pytorch_model.bin")
        print("model dir:", model_path)
        if os.path.exists(model_path):
            pretrained_model = torch.load(model_path)
            pretrained_state_dict = pretrained_model.state_dict()
            filtered_state_dict = {}
            for key, value in pretrained_state_dict.items():
                new_key = key.replace("backbone.", "")
                filtered_state_dict[new_key] = value
            model.load_state_dict(filtered_state_dict, strict=False)
        else:
            print(f"Warning: Pretrained model not found at {model_path}")
    logger.info(f"Training model from existing model file pytorch_model.bin ")

    

    eval_dataset = None
    if training_args.do_eval:
        print("load eval_dataset from ", other_args.valid_file)
        eval_dataset = load_dataset("csv", data_files={"d": other_args.valid_file})["d"]

    train_dataset = None
    if training_args.do_train:
        print("load train_dataset from ", other_args.train_file)
        train_dataset = load_dataset("csv", data_files={"d": other_args.train_file})[
            "d"
        ]

    data_collator = DataCollator(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        metrics = train_result.metrics
        metrics["train_samples"] = len(train_dataset)
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        print("traning end")

    if training_args.do_eval and other_args.test_file:
        for test_file in other_args.test_file:
            test_dataset = load_dataset("csv", data_files={"d": test_file})["d"]
            eval_result = trainer.evaluate(eval_dataset=test_dataset)
            with open("evaluation_results.txt", "a") as file:
                file.write(str(eval_result) + "\t" + training_args.output_dir + "\n")
            print("test")
            print(eval_result)


if __name__ == "__main__":
    main()
