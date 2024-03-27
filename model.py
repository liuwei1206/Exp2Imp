# author = liuwei
# date = 2022-09-06

import math
import os
import json

import numpy as np
import torch
import random
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.activations import gelu
from transformers import PreTrainedModel
from transformers.models.bert import BertModel, BertForMaskedLM
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaForMaskedLM

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import pearsonr

import warnings
warnings.filterwarnings("ignore")

class BaseClassifier(PreTrainedModel):
    def __init__(self, config, args):
        super(BaseClassifier, self).__init__(config)
        if args.encoder_type.lower() == "bert":
            self.encoder = BertModel.from_pretrained(args.model_name_or_path, config=config)
        elif args.encoder_type.lower() == "roberta":
            self.encoder = RobertaModel.from_pretrained(args.model_name_or_path, config=config)

        self.dropout = nn.Dropout(p=config.HP_dropout)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.num_labels = args.num_labels

    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        labels=None,
        flag="Train"
    ):
        encoder_output = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        pooler_output = encoder_output.pooler_output
        pooler_output = self.dropout(pooler_output)
        logits = self.classifier(pooler_output)
        probs = F.softmax(logits, dim=-1)
        target_probs = torch.gather(probs, dim=1, index=labels.unsqueeze(1))
        _, preds = torch.max(logits, dim=-1)
        outputs = (preds, target_probs,)

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss, ) + outputs

        return outputs

    def get_sent_vectors(
        self,
        input_ids,
        attention_mask,
        token_type_ids
    ):
        with torch.no_grad():
            encoder_output = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        pooler_output = encoder_output.pooler_output

        return pooler_output


class TwoEncoder(PreTrainedModel):
    def __init__(self, config, args):
        super(TwoEncoder, self).__init__(config)

        self.encoder_type = args.encoder_type.lower()
        self.is_hard = args.is_hard
        if self.encoder_type == "bert":
            self.encoder = BertForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
        elif self.encoder_type == "roberta":
            self.encoder = RobertaForMaskedLM.from_pretrained(args.model_name_or_path, config=config)
        self.classifier = nn.Linear(config.hidden_size, args.num_labels)
        self.dropout = nn.Dropout(p=config.HP_dropout)
        self.num_connectives = args.num_connectives
        self.num_labels = args.num_labels
        self.conn_onehot_in_vocab = args.conn_onehot_in_vocab # [conn_num, vocab_size]
        self.conn_length_in_vocab = args.conn_length_in_vocab # [conn_num]


    def forward(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        mask_position_ids,
        conn_ids=None,
        labels=None,
        flag="Train"
    ):
        """
        batch_size: N
        seq_length: L
        hidden_size: D
        Args:
            input_ids: [N, L], args1 [mask] args2
            attention_mask: [N, L], 哪些位置是有效的
            mask_position_ids: [N], the position of [mask] tokens
            conn_ids: [N], ground truth connective ids
            labels: [N], relation labels
        """
        batch_size = input_ids.size(0)
        seq_length = input_ids.size(1)

        ## 1 for connective prediction
        if self.encoder_type == "bert":
            conn_output = self.encoder.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        else:
            conn_output = self.encoder.roberta(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        last_hidden_states = conn_output.last_hidden_state # [N, L, D]
        hidden_size = last_hidden_states.size(2)

        # 1.2 scatter the mask position
        mask_position_index = mask_position_ids.view(-1, 1, 1) # [N, 1, 1]
        mask_position_index = mask_position_index.repeat(1, 1, hidden_size) # [N, 1, D]
        mask_token_states = torch.gather(last_hidden_states, dim=1, index=mask_position_index) # [N, 1, D]
        mask_token_states = mask_token_states.squeeze() # [N, D]

        # 1.3 make use of masked_language_linear function
        if self.encoder_type == "bert":
            mask_token_states = self.encoder.cls.predictions.transform(mask_token_states)
            conn_decoder_weight = torch.matmul(self.conn_onehot_in_vocab, self.encoder.cls.predictions.decoder.weight)
            conn_decoder_bias = torch.matmul(self.conn_onehot_in_vocab, self.encoder.cls.predictions.decoder.bias.unsqueeze(1))
            conn_embeddings = torch.matmul(self.conn_onehot_in_vocab, self.encoder.bert.embeddings.word_embeddings.weight)
        elif self.encoder_type == "roberta":
            mask_token_states = self.encoder.lm_head.dense(mask_token_states)
            mask_token_states = gelu(mask_token_states)
            mask_token_states = self.encoder.lm_head.layer_norm(mask_token_states) # [N, D]
            conn_decoder_weight = torch.matmul(self.conn_onehot_in_vocab, self.encoder.lm_head.decoder.weight)  # [conn_num, D]
            conn_decoder_bias = torch.matmul(self.conn_onehot_in_vocab, self.encoder.lm_head.decoder.bias.unsqueeze(1))  # [conn_num, 1]
            conn_embeddings = torch.matmul(self.conn_onehot_in_vocab, self.encoder.roberta.embeddings.word_embeddings.weight)
        conn_decoder_weight = conn_decoder_weight / self.conn_length_in_vocab.unsqueeze(1)
        conn_decoder_bias = conn_decoder_bias / self.conn_length_in_vocab.unsqueeze(1)
        conn_embeddings = conn_embeddings / self.conn_length_in_vocab.unsqueeze(1)
        conn_decoder_weight = torch.transpose(conn_decoder_weight, 1, 0)  # [D, conn_num]
        conn_decoder_bias = torch.transpose(conn_decoder_bias, 1, 0)  # [1, conn_num]
        conn_logits = torch.matmul(mask_token_states, conn_decoder_weight) + conn_decoder_bias # [N, conn_num]

        # 1.4 prepare conn embedding
        if self.is_hard:
            if self.training: # for training, we use gumble-softmax
                conn_scores = F.gumbel_softmax(conn_logits, tau=1.0, hard=True, dim=-1) # [N, CN]
            else: # for evaluation, we use argmax
                conn_scores = torch.argmax(conn_logits, dim=-1) # [N]
                # conn_scores = conn_ids
                ones = torch.eye(self.num_connectives).to(conn_scores.device) # to one-hot
                conn_scores = ones.index_select(dim=0, index=conn_scores) # [N, CN]
        else: # we use softmax, and weighted-sum
            conn_scores = F.softmax(conn_logits, dim=-1) # [N, CN]
        predict_embeds = torch.matmul(conn_scores, conn_embeddings)  # [N, D], a soft connective embedding
        predict_embeds = predict_embeds.unsqueeze(1)  # [N, 1, D]

        ## 2 for relation classifiction
        # 2.1 prepare embeddings, and pass encoder
        if self.encoder_type == "bert":
            input_word_embeds = self.encoder.bert.embeddings.word_embeddings(input_ids)
            input_word_embeds = torch.scatter(input_word_embeds, dim=1, index=mask_position_index, src=predict_embeds)
            rel_output = self.encoder.bert(
                inputs_embeds=input_word_embeds,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        elif self.encoder_type == "roberta":
            input_word_embeds = self.encoder.roberta.embeddings.word_embeddings(input_ids)
            input_word_embeds = torch.scatter(input_word_embeds, dim=1, index=mask_position_index, src=predict_embeds)
            rel_output = self.encoder.roberta(
                inputs_embeds=input_word_embeds,
                attention_mask=attention_mask,
            )

        pooler_output = rel_output.last_hidden_state[:, 0, :]
        pooler_output = self.dropout(pooler_output)
        rel_logits = self.classifier(pooler_output)
        conn_preds = torch.argmax(conn_logits, dim=1)
        rel_preds = torch.argmax(rel_logits, dim=1)
        outputs = (conn_preds, rel_preds,)

        if flag.lower() == "train":
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            conn_loss = loss_fct(conn_logits.view(-1, self.num_connectives), conn_ids.view(-1))
            rel_loss = loss_fct(rel_logits.view(-1, self.num_labels), labels.view(-1))
            loss = conn_loss + rel_loss
            outputs = (loss, conn_loss, rel_loss, ) + outputs

        return outputs
