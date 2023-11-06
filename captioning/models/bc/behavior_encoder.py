import torch
import torch.nn as nn
from torchvision.transforms import InterpolationMode
import numpy as np
import math
import config as CFG
from dataset import AttrDict 


class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=512,
        dropout=0.1
    ):
        super().__init__()
        self.projection = nn.Linear(embedding_dim, projection_dim)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(projection_dim, projection_dim)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(projection_dim)
    
    def forward(self, x):
        projected = self.projection(x)
        x = self.gelu(projected)
        x = self.fc(x)
        x = self.dropout(x)
        x = x + projected
        x = self.layer_norm(x)
        return x

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)

        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class BehaviourEncoder(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.pos_encoder = PositionalEncoding(CFG.d_model, dropout=CFG.dropout)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=CFG.d_model, nhead=CFG.n_heads, batch_first=True, dim_feedforward=CFG.d_ff, dropout=CFG.dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=CFG.n_layers)
        
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        
        self.d_model = CFG.d_model
        self.sequence_projection_head = ProjectionHead(embedding_dim=CFG.d_model, projection_dim=CFG.projection_dim)
        self.text_projection_head = ProjectionHead(embedding_dim=CFG.text_embed_dim, projection_dim=CFG.projection_dim)
        # define a cls token
        self.CLS = nn.Parameter(torch.randn(1, 1, CFG.d_model))
        self.CLS.requires_grad = True

        # self.image_encoder = ImageEncoder(img_dim=CFG.img_size, hidden_dim=CFG.vision_embed_dim)


        # for param in self.clip_image_encoder.parameters():
        #     param.requires_grad = False

        # for param in self.clip_text_encoder.parameters():
        #     param.requires_grad = False
        

        # self.text_final = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(), nn.Linear(1024, 512))
        # self.image_action_encoder = nn.Sequential(nn.Linear(514, 1024), nn.ReLU(), nn.Linear(1024, 512)) # clip model
        #self.image_action_encoder = nn.Sequential(nn.Linear(2050, 1024), nn.ReLU(), nn.Linear(1024, 512)) # r3m model

        # self.decoder = nn.Sequential(nn.Linear(CFG.projection_dim + CFG.vision_embed_dim, 512),
        #                              nn.LeakyReLU(),
        #                              nn.Linear(512, 2))
        #                             #  nn.LeakyReLU(),
        #                             #  nn.Linear(128, 2),
        #                             #  nn.Tanh())

        self.decoder = nn.Sequential(nn.Linear(CFG.projection_dim + CFG.vision_embed_dim, 512),
                                     nn.GELU(), nn.Linear(512, 128),
                                     nn.GELU(), nn.Linear(128, 64),
                                     nn.GELU(), nn.Linear(64, 2),
                                     nn.Tanh())
        
    #     self.lstm = nn.LSTM(input_size = CFG.lstm_input_dim,
    #                         hidden_size = CFG.lstm_hidden_dim,
    #                         num_layers = CFG.lstm_n_layers, 
    #                         batch_first = True)
        
    # def init_hidden(self, batch_size):
    #     hidden_state = torch.zeros(CFG.lstm_n_layers,batch_size,CFG.lstm_hidden_dim)
    #     cell_state = torch.zeros(CFG.lstm_n_layers,batch_size,CFG.lstm_hidden_dim)
    #     self.lstm_hidden = (hidden_state.to(self.device), cell_state.to(self.device))

    def run_decode_batch(self,x,actions,mask,fn):
        batch_size, seq_len, sz = x.shape
        x = x.view(batch_size*seq_len, sz)
        mask = mask.view(batch_size*seq_len)
        x = x[~mask]
        gt_actions = actions.view(batch_size*seq_len, -1)
        gt_actions = gt_actions[~mask]
        actions_recon = fn(x)
        return actions_recon, gt_actions.detach()
        
    
    def forward(self, src):

        # pass video sequence through image encoder in a for loop
        #image_features = []
        #for o in src.observations:
        #    image_features.append(self.image_encoder(o.to(CFG.device)))

        #image_features = pad_sequence(image_features, batch_first=True, padding_value=0.0)

        image_features = src.observations # images: [batch_size, sequence_length, 2048]
        actions = src.actions
        #instructions = src.instruction
        # Process frozen text features with final NN layer
        # text_encoding = self.text_final(src.instruction.to(torch.float))
        text_encoding = src.instruction.to(torch.float)
        text_encoding = self.text_projection_head(text_encoding)
        
        # features = self.process_images(images) # features: [batch_size, sequence_length, 512]
        # concatenate features with actions
        # image_features = torch.cat((features, actions), dim=-1)

        features = torch.cat((image_features, actions), dim=-1)
        # add cls token

        """
        print("actions: ", actions.shape)
        print("observations: ", image_features.shape)
        print("features: ", features.shape)
        print("features[0]: ", features.shape[0])
        print("CLS: ", self.CLS.shape)
        print("test: ", self.CLS.repeat(features.shape[0], 1, 1).shape)
        """
        

        features = torch.cat((self.CLS.repeat(features.shape[0], 1, 1), features), dim=1)
        src_key_padding_mask = (image_features.mean(dim=2)==0.0)
        # include cls token in mask
        src_key_padding_mask = torch.cat((torch.zeros(src_key_padding_mask.shape[0], 1).to(self.device).bool(), src_key_padding_mask), dim=1)

        # Transformer encoder
        # add positional encoding
        features = features * math.sqrt(self.d_model)
        features = self.pos_encoder(features)
        behaviour_encoding = self.transformer_encoder(features, src_key_padding_mask=src_key_padding_mask)
        behaviour_encoding = behaviour_encoding[:, 0, :]
        behaviour_encoding = self.sequence_projection_head(behaviour_encoding)


        # # LSTM encoder
        # # compute length of sequence of each batch element
        # temp = ~src_key_padding_mask
        # seq_lens = temp.sum(dim=1).detach().cpu().numpy()
        # # LSTM encoder
        # X_features = torch.nn.utils.rnn.pack_padded_sequence(features, seq_lens, batch_first=True, enforce_sorted=False)
        # self.init_hidden(image_features.shape[0])
        # X_features, self.lstm_hidden = self.lstm(X_features, self.lstm_hidden)
        # X_features, _ = torch.nn.utils.rnn.pad_packed_sequence(X_features, batch_first=True)
        # behaviour_encoding = self.lstm_hidden[0].permute(1,0,2)[:,0,:]
        # behaviour_encoding = self.sequence_projection_head(behaviour_encoding)


        # Decode actions
        # concatenate the plan to the state
        # the behaviour encoding is our plan
        # WIP TODO test with using the text encoding as plan




        # batch_size, seq_len, sz = image_features.shape
        # be_tiled = behaviour_encoding.repeat(1,seq_len).view(batch_size, seq_len,sz)
        # policy_inputs = torch.cat((image_features, be_tiled), 2)
        # padding_mask = (image_features.mean(dim=2)==0.0)
        # actions_recon, gt_actions = self.run_decode_batch(policy_inputs, actions, padding_mask, self.decoder)

        # normalized features
        behaviour_encoding_norm = behaviour_encoding / behaviour_encoding.norm(dim=1, keepdim=True)
        text_encoding_norm = text_encoding / text_encoding.norm(dim=1, keepdim=True)

        # Skill Generator
        #batch_size, seq_len, sz = image_features.shape
        #te_tiled = text_encoding_norm.repeat(1,seq_len).view(batch_size, seq_len,sz)
        #policy_inputs = torch.cat((image_features, te_tiled), 2)
        #padding_mask = (image_features.mean(dim=2)==0.0)
        #actions_recon, gt_actions = self.run_decode_batch(policy_inputs.detach(), actions, padding_mask, self.decoder)

        # TODO WIP
        #logit_scale = self.logit_scale
        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_sequence = logit_scale.detach() * behaviour_encoding_norm @ text_encoding_norm.t()
        logits_per_text = logits_per_sequence.t()

        return AttrDict(behaviour_encoding=behaviour_encoding_norm, 
                        text_encoding=text_encoding_norm, 
                        logits_per_sequence=logits_per_sequence, 
                        logits_per_text=logits_per_text,
                        #actions_recon=actions_recon,
                        #gt_actions=gt_actions,
                        logit_scale=logit_scale)