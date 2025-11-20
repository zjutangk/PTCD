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
from datasets import load_dataset, Dataset, concatenate_datasets, load_metric
from torch import nn
import codecs
import copy


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
    DataCollatorForLanguageModeling,
    BertForMaskedLM,
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
    warm_train_file: List[str]
    warm_valid_file: List[str]
    num_warm_epochs: int
    train_file: List[str]
    valid_file: List[str]
    warm_learning_rate: float
    num_labels: int
    warm_output_dir: str
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
        # print(anchor_inputs)#input_ids  token_type_ids    attention_mask
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


def data_token(examples, tokenizer, max_input_length):
   
    texts = []
    for example in examples:
        if isinstance(example, dict) and "sentence" in example:
            text = str(example["sentence"])  
            texts.append(text)
        else:
            print(f"Warning: Skipping invalid example: {example}")

    
    if not texts:
        raise ValueError("No valid texts found in examples")

    try:
        batch = tokenizer(
            texts,
            max_length=max_input_length,
            padding=True,
            truncation=True,
            return_special_tokens_mask=True,
        )
    except Exception as e:
        print(f"Tokenization error: {e}")
        print(f"First few texts: {texts[:2]}")
        raise

    dataset = Dataset.from_dict(batch)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    return dataset


def main():
    parser = HfArgumentParser((TrainingArguments, OtherArguments))
    training_args, other_args = parser.parse_args_into_dataclasses()
    training_args.dataloader_prefetch_factor = None
    
    logger.info(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    logger.info(f"other_args {other_args}")

    set_seed(training_args.seed)

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

    warm_training_args = TrainingArguments(
        output_dir=other_args.warm_output_dir,
        evaluation_strategy=training_args.evaluation_strategy,
        overwrite_output_dir=training_args.overwrite_output_dir,
        num_train_epochs=other_args.num_warm_epochs,
        learning_rate=other_args.warm_learning_rate,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        logging_steps=training_args.logging_steps,
        dataloader_prefetch_factor=None,
        dataloader_num_workers=0,
    )

    tokenizer = AutoTokenizer.from_pretrained(other_args.model_dir, use_fast=True)
    if os.path.exists(warm_training_args.output_dir):
        config = AutoConfig.from_pretrained(other_args.model_dir)
        config.num_labels = other_args.num_labels
        warm_model = BertForMaskedLM.from_pretrained(
            warm_training_args.output_dir, config=config
        )
        warm_trainer_model = warm_model.bert
    else:
        warm_train_dataset = load_dataset(
            "csv", data_files={"d": other_args.warm_train_file}
        )["d"]
        warm_eval_dataset = load_dataset(
            "csv", data_files={"d": other_args.warm_valid_file}
        )["d"]

        config = AutoConfig.from_pretrained(other_args.model_dir)
        config.num_labels = other_args.num_labels
        warm_model = BertForMaskedLM.from_pretrained(
            other_args.model_dir, config=config
        )
        warm_data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=True, mlm_probability=0.2
        )
        warm_train_dataset = data_token(warm_train_dataset, tokenizer, 512)
        warm_eval_dataset = data_token(warm_eval_dataset, tokenizer, 512)

        warm_trainer = Trainer(
            model=warm_model,
            args=warm_training_args,
            train_dataset=warm_train_dataset,
            eval_dataset=warm_eval_dataset,
            data_collator=warm_data_collator,
        )

        if other_args.num_warm_epochs != 0:
            warm_train_result = warm_trainer.train()
            warm_metrics = warm_train_result.metrics
            warm_metrics["warm_train_samples"] = len(warm_train_dataset)
            warm_trainer.log_metrics("warm_train", warm_metrics)
            warm_trainer.save_metrics("warm_train", warm_metrics)
            warm_trainer.save_state()
            warm_trainer.save_model()

        warm_trainer_model = warm_trainer.model.bert

    print("-----------------------begin training----------------------------")
    print(
        f"-----------------------config.model_type:{config.model_type}----------------------------"
    )

    model = BertForTripleTextEncoding.from_pretrained(
        other_args.model_dir, config=config
    )
    model.bert.load_state_dict(warm_trainer_model.state_dict(), strict=False)
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
        trainer.save_model()  
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
            print(eval_result)


if __name__ == "__main__":
    main()
