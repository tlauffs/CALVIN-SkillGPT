import numpy as np
import math
from transformers import GPT2LMHeadModel
import numpy as np
import numpy as np
from enum import Enum
from typing import Optional
import torch
import torch.nn as nn
from utils.utils import AttrDict
import config as CFG


class MappingType(Enum):
    MLP = 'mlp'
    Transformer = 'transformer'


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        #pe[0, :, 1::2] = torch.cos(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term[:-1])

        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class BehaviourEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pos_encoder = PositionalEncoding(CFG.d_model, dropout=CFG.dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=CFG.d_model, nhead=CFG.n_heads, batch_first=True, dim_feedforward=CFG.d_ff, dropout=CFG.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=CFG.n_layers)

    def forward(self, src):

        image_features = src.observations # images: [batch_size, sequence_length, 2048]
        actions = src.actions
        #state = src.state

        src_key_padding_mask = (image_features.mean(dim=2)==0.0)
        features = torch.cat((image_features, actions), dim=-1)
        #features = torch.cat((image_features, actions, state), dim=-1)

        # Transformer encoder
        # add positional encoding
        features = features * math.sqrt(CFG.d_model)
        features = self.pos_encoder(features)
        behaviour_encoding = self.transformer_encoder(features, src_key_padding_mask=src_key_padding_mask)

        #print("behaviour_encoding: ", behaviour_encoding.shape)

        return behaviour_encoding

class TransformerMapper(nn.Module):

    def __init__(self, dim_clip: int, dim_embedding: int, prefix_length: int, clip_length: int, num_layers: int = 8):
        super(TransformerMapper, self).__init__()
        self.clip_length = clip_length
        #self.transformer = Transformer(dim_embedding, 8, num_layers)

        self.encoder_layer = nn.TransformerEncoderLayer(d_model=dim_embedding, nhead=8, batch_first=True, dim_feedforward=int(dim_embedding * 2), dropout=0.0)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.linear = nn.Linear(dim_clip, clip_length * dim_embedding)
        self.prefix_const = nn.Parameter(torch.randn(prefix_length, dim_embedding), requires_grad=True)

    def forward(self, x):
        x = self.linear(x).view(x.shape[0], self.clip_length, -1)
        prefix = self.prefix_const.unsqueeze(0).expand(x.shape[0], *self.prefix_const.shape)
        prefix = torch.cat((x, prefix), dim=1)
        out = self.transformer(prefix)[:, self.clip_length:]
        return out

class ClipCaptionModel(nn.Module):

    def __init__(self, prefix_length: int, clip_length: Optional[int] = None, prefix_size: int = 512,
                 num_layers: int = 8, mapping_type: MappingType = MappingType.MLP):
        


        super(ClipCaptionModel, self).__init__()
        self.prefix_length = prefix_length
        self.gpt = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt_embedding_size = self.gpt.transformer.wte.weight.shape[1]
        self.clip_project = TransformerMapper(prefix_size, self.gpt_embedding_size, prefix_length,
                                                                     clip_length, num_layers)
        self.behaviour_encoder = BehaviourEncoder()
        # self.project_to_gpt = nn.Linear(514, self.gpt_embedding_size).to("cuda")
        self.project_to_gpt = nn.Linear(CFG.d_model, self.gpt_embedding_size).to("cuda")

    def get_dummy_token(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, self.prefix_length, dtype=torch.int64, device=device)

    def forward(self, data: AttrDict):

        tokens = data.gpt_tokens
        gpt_mask = data.gpt_mask
        labels = None

        embedding_text = self.gpt.transformer.wte(tokens)
        behaviour_encoding  = self.behaviour_encoder(data)
        
        behaviour_encoder_padding_mask = ~(data.observations.mean(dim=2)==0.0) * 1.0
    
        prefix_projections = self.project_to_gpt(behaviour_encoding)
        total_mask = torch.cat((behaviour_encoder_padding_mask, gpt_mask), dim=1)

        # prefix_projections = self.clip_project(prefix).view(-1, self.prefix_length, self.gpt_embedding_size)
        embedding_cat = torch.cat((prefix_projections, embedding_text), dim=1)
        if labels is not None:
            dummy_token = self.get_dummy_token(tokens.shape[0], tokens.device)
            labels = torch.cat((dummy_token, tokens), dim=1)
        out = self.gpt(inputs_embeds=embedding_cat, labels=labels, attention_mask=total_mask)
        return out

class ClipCaptionPrefix(ClipCaptionModel):

    def parameters(self, recurse: bool = True):
        return self.clip_project.parameters()

    def train(self, mode: bool = True):
        super(ClipCaptionPrefix, self).train(mode)
        self.gpt.eval()
        return self