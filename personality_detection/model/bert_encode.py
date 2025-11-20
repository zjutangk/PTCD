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

Temperature = 0.05
TopK = 5
Margin = 0.6
# Margin = 0.8
Poi = 0.15
Neg = 1.7


class TripletMarginLoss_changed:
    def __init__(self, margin):
        self.margin = margin

    def compute_loss(self, anchor, positive, negative):
        pos_distance = torch.norm(anchor - positive, p=2, dim=1)  # shape: N
        # changed
        neg_distance = torch.norm(anchor - negative, p=2, dim=1)  # shape: N

        losses = torch.maximum(
            pos_distance - neg_distance + self.margin, torch.tensor(0.0)
        )
        losses = pos_distance - neg_distance + self.margin + pos_distance

        return losses.mean()


class BertForTripleTextEncoding(BertPreTrainedModel):
    def __init__(self, config=None, model=None):
        super().__init__(config)
        if model is None:
            self.bert = BertModel(config)
        else:
            self.bert = model
        self.loss_fuc1 = nn.TripletMarginLoss(margin=Margin)
        self.loss_fuc2 = TripletMarginLoss_changed(margin=Margin)
        self.loss_cross = nn.CrossEntropyLoss()
        self.init_weights()
  
    def forward(
        self,
        anchor_input_ids=None,
        anchor_attention_mask=None,
        anchor_token_type_ids=None,
        positive_input_ids=None,
        positive_attention_mask=None,
        positive_token_type_ids=None,
        negative_input_ids=None,
        negative_attention_mask=None,
        negative_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):
        # print(anchor_input_ids)
        anchor_outputs = self.bert(
            input_ids=anchor_input_ids,
            attention_mask=anchor_attention_mask,
            token_type_ids=anchor_token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        positive_outputs = self.bert(
            input_ids=positive_input_ids,
            attention_mask=positive_attention_mask,
            token_type_ids=positive_token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        negative_outputs = self.bert(
            input_ids=negative_input_ids,
            attention_mask=negative_attention_mask,
            token_type_ids=negative_token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        anchor_embedding = anchor_outputs.hidden_states[-1][:,0]
        positive_embedding = positive_outputs.hidden_states[-1][:,0]
        negative_embedding = negative_outputs.hidden_states[-1][:,0]
        anchor_embedding = F.normalize(anchor_embedding, p=2, dim=1)
        positive_embedding = F.normalize(positive_embedding, p=2, dim=1)
        negative_embedding = F.normalize(negative_embedding, p=2, dim=1)

        loss = self.loss_fuc1(anchor_embedding, positive_embedding, negative_embedding)

        poi_distance = torch.norm(
            anchor_embedding - positive_embedding, dim=1, p=2, keepdim=True
        )
        neg_distance = torch.norm(
            anchor_embedding - negative_embedding, dim=1, p=2, keepdim=True
        )
        l2_margin = neg_distance - poi_distance  #

        return {"loss": loss, "l2_margin": l2_margin}

    def get_feature(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=True,
        return_dict=None,
    ):
        # print(anchor_input_ids)
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        embedding = outputs.hidden_states[-1][:,0]
        print("embedding.shape",embedding.shape)
    
        embedding = F.normalize(embedding, p=2, dim=1)
       
        return embedding



class BertForEmbedding(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        embedding = outputs[1]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding


def encoder_forward(
    encoder,
    anchor_input_ids=None,
    anchor_attention_mask=None,
    anchor_token_type_ids=None,
    positive_input_ids=None,
    positive_attention_mask=None,
    positive_token_type_ids=None,
    negative_input_ids=None,
    negative_attention_mask=None,
    negative_token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    output_attentions=None,
    output_hidden_states=None,
    return_dict=None,
):
    anchor_outputs_0 = encoder(
        input_ids=anchor_input_ids,
        attention_mask=anchor_attention_mask,
        token_type_ids=anchor_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
    )
    positive_outputs_0 = encoder(
        input_ids=positive_input_ids,
        attention_mask=positive_attention_mask,
        token_type_ids=positive_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
    )
    negative_outputs_0 = encoder(
        input_ids=negative_input_ids,
        attention_mask=negative_attention_mask,
        token_type_ids=negative_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
    )

    anchor_outputs_1 = encoder(
        input_ids=anchor_input_ids,
        attention_mask=anchor_attention_mask,
        token_type_ids=anchor_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
    )
    positive_outputs_1 = encoder(
        input_ids=positive_input_ids,
        attention_mask=positive_attention_mask,
        token_type_ids=positive_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
    )
    negative_outputs_1 = encoder(
        input_ids=negative_input_ids,
        attention_mask=negative_attention_mask,
        token_type_ids=negative_token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        output_attentions=output_attentions,
        output_hidden_states=True,
        return_dict=return_dict,
    )
    return (
        anchor_outputs_0[0][:, 0],
        positive_outputs_0[0][:, 0],
        negative_outputs_0[0][:, 0],
        anchor_outputs_1[0][:, 0],
        positive_outputs_1[0][:, 0],
        negative_outputs_1[0][:, 0],
    )


def similarity_loss(emb1, emb2):
    sim_matrix = F.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)
    labels = torch.arange(sim_matrix.size(0)).long().to(sim_matrix.device)
    sim_matrix = sim_matrix / Temperature
    loss = F.cross_entropy(sim_matrix, labels)
    return loss


def neighbor_loss(emb1, emb2):
    similarity_matrix = torch.einsum("nd,md->nm", emb1, emb2) / 0.1
    topk_values, topk_indices = torch.topk(similarity_matrix, k=TopK, dim=1)
    mask = torch.zeros_like(similarity_matrix)
    mask.scatter_(1, topk_indices, 1)
    similarity_matrix_mask = torch.exp(similarity_matrix) * mask
    sim = (
        similarity_matrix_mask.sum(dim=1) / (similarity_matrix.sum(dim=1) + 1e-6) / TopK
    )
    log_sim = -torch.log(sim)
    return log_sim.sum()


class RobertaForTripleTextEncoding(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.loss_fuc = nn.TripletMarginLoss()
        self.init_weights()

    def forward(
        self,
        anchor_input_ids=None,
        anchor_attention_mask=None,
        anchor_token_type_ids=None,
        positive_input_ids=None,
        positive_attention_mask=None,
        positive_token_type_ids=None,
        negative_input_ids=None,
        negative_attention_mask=None,
        negative_token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        anchor_0, positive_0, negative_0, anchor_1, positive_1, negative_1 = (
            encoder_forward(
                self.roberta,
                anchor_input_ids=anchor_input_ids,
                anchor_attention_mask=anchor_attention_mask,
                anchor_token_type_ids=anchor_token_type_ids,
                positive_input_ids=positive_input_ids,
                positive_attention_mask=positive_attention_mask,
                positive_token_type_ids=positive_token_type_ids,
                negative_input_ids=negative_input_ids,
                negative_attention_mask=negative_attention_mask,
                negative_token_type_ids=negative_token_type_ids,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        )

        merge_0 = torch.cat((anchor_0, positive_0, negative_0), dim=0)
        merge_1 = torch.cat((anchor_1, positive_1, negative_1), dim=0)
        anchor_0 = F.normalize(anchor_0, p=2, dim=1)
        positive_0 = F.normalize(positive_0, p=2, dim=1)
        negative_0 = F.normalize(negative_0, p=2, dim=1)
        anchor_1 = F.normalize(anchor_0, p=2, dim=1)
        positive_1 = F.normalize(positive_0, p=2, dim=1)
        negative_1 = F.normalize(negative_0, p=2, dim=1)

        loss = self.loss_fuc(anchor_0, positive_0, negative_0)
        poi_distance = torch.norm(anchor_0 - positive_0, dim=1, p=2, keepdim=True)
        neg_distance = torch.norm(anchor_0 - negative_0, dim=1, p=2, keepdim=True)
        l2_margin = neg_distance - poi_distance  #

        merge_0 = F.normalize(merge_0, p=2, dim=1)
        merge_1 = F.normalize(merge_1, p=2, dim=1)
        loss += similarity_loss(merge_0, merge_1)
        # loss += neighbor_loss(merge_0, merge_1)

        return {"loss": loss, "l2_margin": l2_margin}


class RobertaForEmbedding(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        next_sentence_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        embedding = outputs[0][:, 0]
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding

