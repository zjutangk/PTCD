import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from utils.tools import *


class MAB(nn.Module):
    def __init__(self, dim_Q, dim_K, dim_V, num_heads, ln=False):
        super(MAB, self).__init__()
        self.dim_V = dim_V
        self.num_heads = num_heads
        self.fc_q = nn.Linear(dim_Q, dim_V)
        self.fc_k = nn.Linear(dim_K, dim_V)
        self.fc_v = nn.Linear(dim_K, dim_V)
        if ln:
            self.ln0 = nn.LayerNorm(dim_V)
            self.ln1 = nn.LayerNorm(dim_V)
        self.fc_o = nn.Linear(dim_V, dim_V)

    def forward(self, Q, K):
        Q = self.fc_q(Q)
        K, V = self.fc_k(K), self.fc_v(K)

        dim_split = self.dim_V // self.num_heads
        Q_ = torch.cat(Q.split(dim_split, 2), 0)
        K_ = torch.cat(K.split(dim_split, 2), 0)
        V_ = torch.cat(V.split(dim_split, 2), 0)

        A = torch.softmax(Q_.bmm(K_.transpose(1, 2)) / math.sqrt(self.dim_V), 2)
        O = torch.cat((Q_ + A.bmm(V_)).split(Q.size(0), 0), 2)
        O = O if getattr(self, "ln0", None) is None else self.ln0(O)
        O = O + F.relu(self.fc_o(O))
        O = O if getattr(self, "ln1", None) is None else self.ln1(O)
        return O


class SAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, ln=False):
        super(SAB, self).__init__()
        self.mab = MAB(dim_in, dim_in, dim_out, num_heads, ln=ln)

    def forward(self, X):
        return self.mab(X, X)


class ISAB(nn.Module):
    def __init__(self, dim_in, dim_out, num_heads, num_inds, ln=False):
        super(ISAB, self).__init__()
        self.I = nn.Parameter(torch.Tensor(1, num_inds, dim_out))
        nn.init.xavier_uniform_(self.I)
        self.mab0 = MAB(dim_out, dim_in, dim_out, num_heads, ln=ln)
        self.mab1 = MAB(dim_in, dim_out, dim_out, num_heads, ln=ln)

    def forward(self, X):
        H = self.mab0(self.I.repeat(X.size(0), 1, 1), X)
        return self.mab1(X, H)


class PMA(nn.Module):
    def __init__(self, dim, num_heads, num_seeds, ln=False):
        super(PMA, self).__init__()
        self.S = nn.Parameter(torch.Tensor(1, num_seeds, dim))
        nn.init.xavier_uniform_(self.S)
        self.mab = MAB(dim, dim, dim, num_heads, ln=ln)

    def forward(self, X):

        return self.mab(self.S.repeat(X.size(0), 1, 1), X)


import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers import (
    BertModel,
    BertPreTrainedModel,
    RobertaModel,
    RobertaPreTrainedModel,
)
from transformers.modeling_outputs import MaskedLMOutput


class Set_dec_mbti(nn.Module):
    def __init__(self, device=None):
        super(Set_dec_single, self).__init__()

        self.device = device
        dim_hidden = 768
        num_heads = 4
        ln = False
        num_outputs = 1
        dim_output = 128
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, 128, num_heads, ln=ln),
            nn.Dropout(),
        )

        self.classifier = nn.Linear(128, 16)

        self.dec.to(self.device)
        self.classifier.to(self.device)

    def forward(self, embeddings):

        embeddings = embeddings.float()
        pool_embeddings = self.dec(embeddings)
        embeddings = pool_embeddings.squeeze()
        logits = self.classifier(embeddings)
        output_dict = {
            "logits": logits,
        }

        return output_dict

    def loss_bce(self, logits, Y):
        loss = nn.BCEWithLogitsLoss()
        # print(logits.dtype)
        # print(Y.float().dtype)
        output = loss(logits, Y.float())
        return output

    def loss_ce(self, logits, Y):
        loss = nn.CrossEntropyLoss()
        output = loss(logits, Y.float())
        return output


class Set_dec_single(nn.Module):
    def __init__(self, device=None):
        super(Set_dec_single, self).__init__()

        self.device = device
        dim_hidden = 768
        num_heads = 4
        ln = False
        num_outputs = 1
        dim_output = 128
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, 128, num_heads, ln=ln),
            nn.Dropout(),
        )

        self.classifier = nn.Linear(128, 4)

        self.dec.to(self.device)
        self.classifier.to(self.device)

    def forward(self, embeddings):

        embeddings = embeddings.float()
        pool_embeddings = self.dec(embeddings)
        embeddings = pool_embeddings.squeeze()
        logits = self.classifier(embeddings)
        output_dict = {
            "logits": logits,
        }

        return output_dict

    def loss_bce(self, logits, Y):
        loss = nn.BCEWithLogitsLoss()
        output = loss(logits, Y.float())
        return output


class Set_dec_big5(nn.Module):
    def __init__(self, device=None):
        super(Set_dec_big5, self).__init__()
        self.device = device
        dim_hidden = 768
        num_heads = 4
        ln = False
        num_outputs = 5
        dim_output = 128
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_output, num_heads, ln=ln),
            nn.Dropout(),
        )
       
        self.classifier1 = nn.Linear(dim_output, 1)
        
        self.classifier2 = nn.Linear(dim_output, 1)
        
        self.classifier3 = nn.Linear(dim_output, 1)
        
        self.classifier4 = nn.Linear(dim_output, 1)

        self.classifier5 = nn.Linear(dim_output, 1)
        

        # self.bert.to(self.device)
        self.dec.to(self.device)
        self.classifier1.to(self.device)
        self.classifier2.to(self.device)
        self.classifier3.to(self.device)
        self.classifier4.to(self.device)
        self.classifier5.to(self.device)

    def forward(self, embeddings):

        embeddings = embeddings.float()
        # print(embeddings.dtype)
        pool_embeddings = self.dec(embeddings)
        split_embeddings = torch.split(pool_embeddings, 1, dim=1)
        split_embeddings = [t.squeeze() for t in split_embeddings]
        embeddings_1 = split_embeddings[0]
        embeddings_2 = split_embeddings[1]
        embeddings_3 = split_embeddings[2]
        embeddings_4 = split_embeddings[3]
        embeddings_5 = split_embeddings[4]
        logits_1 = self.classifier1(embeddings_1)
        logits_2 = self.classifier2(embeddings_2)
        logits_3 = self.classifier3(embeddings_3)
        logits_4 = self.classifier4(embeddings_4)
        logits_5 = self.classifier5(embeddings_5)
        output_dict = {
            "logits1": logits_1,
            "logits2": logits_2,
            "logits3": logits_3,
            "logits4": logits_4,
            "logits5": logits_5,
        }

        return output_dict

    def loss_bce(self, logits, Y):
        loss = nn.BCEWithLogitsLoss()
        output = loss(logits, Y.float())
        return output

    def loss_bce_ent(self, logits, Y):
        bce_loss = nn.BCEWithLogitsLoss()(logits, Y.float())

        probs = torch.tensor(logits)

        entropy = -torch.mean(
            probs * torch.log(probs + 1e-10)
            + (1 - probs) * torch.log(1 - probs + 1e-10)
        )

        lambda_entropy = 0.1
        total_loss = bce_loss + lambda_entropy * entropy

        return total_loss

    def macro_f1_loss(self, logits, Y):
        probs = torch.Tensor(logits)
        TP = (probs * Y).sum(dim=0)
        FP = (probs * (1 - Y)).sum(dim=0)
        FN = ((1 - probs) * Y).sum(dim=0)

        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)

        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        macro_f1 = f1.mean()
        return 1 - macro_f1 

class Set_dec(nn.Module):
    def __init__(self, device=None):
        super(Set_dec, self).__init__()
        self.device = device
        dim_hidden = 768
        num_heads = 4
        ln = False
        num_outputs = 4
        dim_output = 128
        self.dec = nn.Sequential(
            nn.Dropout(),
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_output, num_heads, ln=ln),
            nn.Dropout(),
        )

        self.classifier1 = nn.Linear(dim_output, 1)
        
        self.classifier2 = nn.Linear(dim_output, 1)
        
        self.classifier3 = nn.Linear(dim_output, 1)
        
        self.classifier4 = nn.Linear(dim_output, 1)

        

        self.dec.to(self.device)
        self.classifier1.to(self.device)
        self.classifier2.to(self.device)
        self.classifier3.to(self.device)
        self.classifier4.to(self.device)

    def forward(self, embeddings):

        embeddings = embeddings.float()
        pool_embeddings = self.dec(embeddings)
        split_embeddings = torch.split(pool_embeddings, 1, dim=1)
        split_embeddings = [t.squeeze() for t in split_embeddings]
        embeddings_1 = split_embeddings[0]
        embeddings_2 = split_embeddings[1]
        embeddings_3 = split_embeddings[2]
        embeddings_4 = split_embeddings[3]
        logits_1 = self.classifier1(embeddings_1)
        logits_2 = self.classifier2(embeddings_2)
        logits_3 = self.classifier3(embeddings_3)
        logits_4 = self.classifier4(embeddings_4)
        output_dict = {
            "logits1": logits_1,
            "logits2": logits_2,
            "logits3": logits_3,
            "logits4": logits_4,
        }

        return output_dict

    def loss_bce(self, logits, Y):
        loss = nn.BCEWithLogitsLoss()
        # print(logits.dtype)
        # print(Y.float().dtype)
        output = loss(logits, Y.float())
        return output

    def loss_bce_ent(self, logits, Y):
        bce_loss = nn.BCEWithLogitsLoss()(logits, Y.float())

        probs = torch.tensor(logits)

        entropy = -torch.mean(
            probs * torch.log(probs + 1e-10)
            + (1 - probs) * torch.log(1 - probs + 1e-10)
        )

        lambda_entropy = 0.1
        total_loss = bce_loss + lambda_entropy * entropy

        return total_loss

    def macro_f1_loss(self, logits, Y):
        probs = torch.Tensor(logits)
        TP = (probs * Y).sum(dim=0)
        FP = (probs * (1 - Y)).sum(dim=0)
        FN = ((1 - probs) * Y).sum(dim=0)
        precision = TP / (TP + FP + 1e-7)
        recall = TP / (TP + FN + 1e-7)

        f1 = 2 * precision * recall / (precision + recall + 1e-7)

        macro_f1 = f1.mean()
        return 1 - macro_f1 


class DeepSet(nn.Module):
    def __init__(self, dim_input, num_outputs, dim_output, dim_hidden=128):
        super(DeepSet, self).__init__()
        self.num_outputs = num_outputs
        self.dim_output = dim_output
        self.enc = nn.Sequential(
            nn.Linear(dim_input, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
        )
        self.dec = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Linear(dim_hidden, num_outputs * dim_output),
        )

    def forward(self, X):
        X = self.enc(X).mean(-2)
        X = self.dec(X).reshape(-1, self.num_outputs, self.dim_output)
        return X


class SetTransformer(nn.Module):
    def __init__(
        self,
        dim_input,
        num_outputs,
        dim_output,
        num_inds=32,
        dim_hidden=128,
        num_heads=4,
        ln=False,
    ):
        super(SetTransformer, self).__init__()
        self.enc = nn.Sequential(
            ISAB(dim_input, dim_hidden, num_heads, num_inds, ln=ln),
            ISAB(dim_hidden, dim_hidden, num_heads, num_inds, ln=ln),
        )
        self.dec = nn.Sequential(
            PMA(dim_hidden, num_heads, num_outputs, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            SAB(dim_hidden, dim_hidden, num_heads, ln=ln),
            nn.Linear(dim_hidden, dim_output),
        )

    def forward(self, X):
        return self.dec(self.enc(X))
