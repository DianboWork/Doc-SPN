import torch.nn as nn
import torch
from models.set_decoder import SetDecoder
from utils.functions import generate_triple
import copy
from transformers import BertModel, BertTokenizer


class SeqEncoder(nn.Module):
    def __init__(self):
        super(SeqEncoder, self).__init__()
        self.bert = BertModel.from_pretrained("/data/suidianbo/pytorch_transformers/Bert/bert_base_cased/")
        self.config = self.bert.config

    def forward(self, input_ids, attention_mask):
        output = self.bert(input_ids, attention_mask=attention_mask)
        return output.last_hidden_state, output.pooler_output

class SetPred4RE(nn.Module):
    def __init__(self):
        super(SetPred4RE, self).__init__()
        self.encoder = SeqEncoder()
        config = self.encoder.config
        self.num_classes = 24
        self.decoder = SetDecoder(config, 15, 3, 24, return_intermediate=False)


    def forward(self, input_ids, attention_mask):
        last_hidden_state, pooler_output = self.encoder(input_ids, attention_mask)
        class_logits, head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = self.decoder(encoder_hidden_states=last_hidden_state, encoder_attention_mask=attention_mask)
        # head_start_logits, head_end_logits, tail_start_logits, tail_end_logits = span_logits.split(1, dim=-1)
        head_start_logits = head_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        head_end_logits = head_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_start_logits = tail_start_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        tail_end_logits = tail_end_logits.squeeze(-1).masked_fill((1 - attention_mask.unsqueeze(1)).bool(), -10000.0)
        outputs = {'pred_rel_logits': class_logits, 'head_start_logits': head_start_logits, 'head_end_logits': head_end_logits, 'tail_start_logits': tail_start_logits, 'tail_end_logits': tail_end_logits} 
        return outputs

    def gen_triples(self, input_ids, attention_mask, info):
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            # print(outputs)
            pred_triple = generate_triple(outputs, info, self.args, self.num_classes)
            # print(pred_triple)
        return pred_triple






