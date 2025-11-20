from utils.tools import *
from model.models import Set_dec
from dataloader_person import *
from init_parameter import init_model
from utils.tools import *
from datetime import datetime
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
import random
import argparse
import logging
import sys
import pickle
import json
from pathlib import Path
import time

num_labels = 4
LABEL_LIST = [
    "ISTJ",
    "ISFJ",
    "INFJ",
    "INTJ",
    "ISTP",
    "ISFP",
    "INFP",
    "INTP",
    "ESTP",
    "ESFP",
    "ENFP",
    "ENTP",
    "ESTJ",
    "ESFJ",
    "ENFJ",
    "ENTJ",
]

# hyperparameters
train_csv_path = ""
test_csv_path = ""  # 方便对比
train_ratio = 0.8
# pretrain_batch_size


def getlogit(model, X):
    logits_1 = model(X)["logits1"]
    logits_2 = model(X)["logits2"]
    logits_3 = model(X)["logits3"]
    logits_4 = model(X)["logits4"]
    if len(logits_1.shape) == 1:
        logits_1 = logits_1.unsqueeze(0)
    if len(logits_2.shape) == 1:
        logits_2 = logits_2.unsqueeze(0)
    if len(logits_3.shape) == 1:
        logits_3 = logits_3.unsqueeze(0)
    if len(logits_4.shape) == 1:
        logits_4 = logits_4.unsqueeze(0)
    logits = torch.cat((logits_1, logits_2, logits_3, logits_4), dim=1)
    # print(logits)
    return logits


class TrainManager:

    def __init__(self, args):
        set_seed(args.seed)
        n_gpu = torch.cuda.device_count()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = Set_dec(device=self.device)
        self.moedel = self.model.float()

        if n_gpu > 1:
            self.model = nn.DataParallel(self.model)

        self.train_data, self.train_dataloader = create_dataloader(
            args.train_csv_file, args.train_batch_size
        )
        self.test_data, self.test_dataloader = create_dataloader(
            args.test_csv_file, args.train_batch_size
        )

        self.num_train_optimization_steps = (
            int(len(self.train_data) / args.train_batch_size) * args.num_train_epochs
        )

        self.optimizer, self.scheduler = self.get_optimizer(args)

        self.best_f1_score = 0
        self.best_acc = 0
        self.best_exact_acc = 0
        self.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.generator = view_generator(self.tokenizer, args.rtr_prob, args.seed)

    def eval_train(self, args):
        """
        calculate acc on validation set
        """
        self.model.eval()

        total_labels = torch.empty((0, num_labels)).to(self.device)
        total_logits = torch.empty((0, num_labels)).to(self.device)

        for batch in tqdm(self.train_dataloader, desc="Iteration"):
            label_ids = batch["labels"].to(self.device)
            features = batch["features"].to(self.device)
            with torch.set_grad_enabled(False):
                
                logits = getlogit(self.model, features)
                # print(logits)
                total_labels = torch.cat((total_labels, label_ids))
                total_logits = torch.cat((total_logits, logits))

        total_probs = torch.sigmoid(total_logits.detach())
        y_pred = total_probs.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        eval_dict = evaluate(y_true, y_pred)

        return eval_dict

    def eval(self, args):
        """
        calculate acc on validation set
        """
        self.model.eval()

        total_labels = torch.empty((0, num_labels)).to(self.device)
        total_logits = torch.empty((0, num_labels)).to(self.device)

        for batch in tqdm(self.test_dataloader, desc="Iteration"):
            
            label_ids = batch["labels"].to(self.device)
            features = batch["features"].to(self.device)
            with torch.set_grad_enabled(False):
                
                logits = getlogit(self.model, features)
                
                total_labels = torch.cat((total_labels, label_ids))
                
                total_logits = torch.cat((total_logits, logits))

        total_probs = torch.sigmoid(total_logits.detach())
       
        y_pred = total_probs.cpu().numpy()
        y_true = total_labels.cpu().numpy()
        eval_dict = evaluate(y_true, y_pred)
       
        return eval_dict

    def train(self, args):
        wait = 0
        best_model = None
        # mlm_iter = iter(self.train_dataloader)  # mlm on semi-dataloader
        for epoch in trange(int(args.num_pretrain_epochs), desc="Epoch"):
            self.model.train()
            tr_loss = 0
            nb_tr_examples, nb_tr_steps = 0, 0
            t1=time.time()
            for step, batch in enumerate(tqdm(self.train_dataloader, desc="Iteration")):
               
                features = batch["features"].to(self.device)
               
                label_ids = batch["labels"].to(self.device)

               
                with torch.set_grad_enabled(True):
                    logits = getlogit(self.model, features)
                    
                    if isinstance(self.model, nn.DataParallel):
                        loss_src = self.model.module.loss_ce(logits, label_ids)
                        
                    else:
                        loss_src = self.model.loss_bce_ent(logits, label_ids)
                        
                    loss_all = loss_src
                    loss_all.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    tr_loss += loss_all.item()

                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    nb_tr_examples += label_ids.size(0)
                    nb_tr_steps += 1
            
            t2=time.time()
            print("time cost",t2-t1)
            loss = tr_loss / nb_tr_steps
            print("train_loss", loss)

           
            results = self.eval_train(args)
            print(f"Epoch {epoch+1},eval_train")
            print(f"Overall Accuracy: {results['overall_acc']:.4f}")
            print(f"Exact Accuracy: {results['exact_accuracy']:.4f}")
            print(f"Macro F1: {results['macro_f1']:.4f}")

           
            for dim in range(4):
                dim_metrics = results["dimension_metrics"][f"dim_{dim}"]
                print(f"\nDimension {dim}:")
                print(f"Accuracy: {dim_metrics['accuracy']:.4f}")
                print(f"F1-Score: {dim_metrics['f1_score']:.4f}")

            results = self.eval(args)
            print(f"Epoch {epoch+1}, eval_test")
            acc = results["overall_acc"]
            exact_acc = results["exact_accuracy"]
            print(f"Overall Accuracy: {results['overall_acc']:.4f}")
            print(f"Exact Accuracy: {results['exact_accuracy']:.4f}")
            print(f"Macro F1: {results['macro_f1']:.4f}")

           
            for dim in range(4):
                dim_metrics = results["dimension_metrics"][f"dim_{dim}"]
                print(f"\nDimension {dim}:")
                print(f"Accuracy: {dim_metrics['accuracy']:.4f}")
                print(f"F1-Score: {dim_metrics['f1_score']:.4f}")

            if acc > self.best_acc:
                best_model = copy.deepcopy(self.model)
                wait = 0
                self.best_acc = acc
                save_path = os.path.join(args.save_path, str(epoch))
                directory = Path(args.save_path)
                if not directory.exists():
                    directory.mkdir(parents=True, exist_ok=True)

               
                torch.save(best_model, save_path)

            if exact_acc > self.best_exact_acc:
                self.best_exact_acc = exact_acc

            else:
                wait += 1
                if wait >= args.wait_patient:
                    break

        self.model = best_model
        print("best_acc", self.best_acc)
        print("best_exact_acc", self.best_exact_acc)

    def compute_kl_loss(self, p, q, pad_mask=None):
        p_loss = F.kl_div(
            F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction="none"
        )
        q_loss = F.kl_div(
            F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction="none"
        )

        # pad_mask is for seq-level tasks
        if pad_mask is not None:
            p_loss.masked_fill_(pad_mask, 0.0)
            q_loss.masked_fill_(pad_mask, 0.0)

        # You can choose whether to use function "sum" and "mean" depending on your task
        p_loss = p_loss.sum()
        q_loss = q_loss.sum()

        loss = (p_loss + q_loss) / 2
        return loss

    def get_optimizer(self, args):
        num_warmup_steps = int(
            args.warmup_proportion * self.num_train_optimization_steps
        )
        param_optimizer = list(self.model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.01,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr_pre)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.num_train_optimization_steps,
        )

        return optimizer, scheduler

   

    def get_features_labels(self, dataloader, model, args):
        model.eval()
        total_features = torch.empty((0, args.feat_dim)).to(self.device)
        total_labels = torch.empty(0, dtype=torch.long).to(self.device)

        for batch in tqdm(dataloader, desc="Extracting representation"):
            batch = tuple(t.to(self.device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch
            X = {
                "input_ids": input_ids,
                "attention_mask": input_mask,
                "token_type_ids": segment_ids,
            }
            with torch.no_grad():
                feature = model(X, output_hidden_states=True)["hidden_states"]

            total_features = torch.cat((total_features, feature))
            total_labels = torch.cat((total_labels, label_ids))

        return total_features, total_labels


if __name__ == "__main__":

    print("Data and Parameters Initialization...")
    parser = init_model()
    args = parser.parse_args()
    print(args)

    manager = TrainManager(args)
    acc = manager.eval(args)
    print(f"before train, acc is {acc}")
    manager.train(args)
    model = manager.model
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    model_name = f"model_{current_time}.pth"
    save_path = os.path.join(args.save_path, model_name)
    torch.save(model.state_dict(), save_path)
