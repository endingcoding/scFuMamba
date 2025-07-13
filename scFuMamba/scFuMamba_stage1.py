import torch
import timm
import numpy as np
import os
import math
from einops import repeat, rearrange
from einops.layers.torch import Rearrange
from torchvision.transforms import ToTensor, Compose, Normalize
from timm.models.layers import trunc_normal_
import anndata as ad
from sklearn.model_selection import train_test_split
import time
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import warnings
from mamba_ssm import Mamba
import torch
import torch.nn.functional as F
from utils import *
torch.autograd.set_detect_anomaly(True)

warnings.filterwarnings("ignore")

class Config(object):
    """para"""
    def __init__(self):
        self.device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')  
        self.dropout = 0.1                                    
        self.num_classes = 18                                             
        self.batch_size = 128                                       
        self.lr = 5e-4                      
        self.encoder_layer = 1
        self.decoder_layer = 1
        self.mask_ratio = 0.15
        
        self.RNA_tokens = 4000 # RNA number
        self.RNA_component = 2000 # channel        
        self.emb_RNA = 2
        
        self.mask_ratio1 = 0.15
        
        self.ADT_tokens = 14 # proteins umber
        self.ADT_component = 14 # channel
        self.emb_ADT = 1 
        
        self.emb_dim = 128       
        self.total_epoch = 500
        self.warmup_epoch = 10

        self.k_size =1

config = Config()

# RNA encoder
class RNA_Encoder(nn.Module):
    def __init__(self, emb_dim=64, emb_RNA=10, RNA_component=400, RNA_tokens=4000, encoder_layer=6, mask_ratio=0.1):
        super().__init__()
        self.RNA_component = RNA_component
        self.emb_RNA = emb_RNA

        self.tokens = nn.Sequential(nn.Linear(in_features=RNA_tokens, out_features=RNA_tokens))
        self.embedding = nn.Sequential(nn.Linear(in_features=emb_RNA, out_features=emb_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, RNA_component, emb_dim))
        self.shuffle = PatchShuffle(mask_ratio)
        self.mamba = nn.Sequential(*[Mamba(d_model=emb_dim) for _ in range(encoder_layer)])
        self.layer_norm = nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, patches):
        patches = self.tokens(patches)
        patches = patches.view(patches.size(0), self.RNA_component, self.emb_RNA)
        patches = self.embedding(patches) 
        patches = patches + self.pos_embedding
        
        patches = rearrange(patches, 'b c s -> c b s')  
        patches, _, backward_indexes = self.shuffle(patches)   
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')

        features = self.layer_norm(self.mamba(patches))
        features = rearrange(features, 'b t c -> t b c')
        return features, backward_indexes


# ADT encoder
class ADT_Encoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_ADT=10,ADT_component=11,ADT_tokens=110, encoder_layer=6,mask_ratio1=0.1
                 )-> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = ADT_tokens, out_features = ADT_tokens))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_ADT, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, ADT_component ,emb_dim)))
        self.shuffle = PatchShuffle(mask_ratio1)
        self.mamba = torch.nn.Sequential(*[Mamba(d_model=emb_dim) for _ in range(encoder_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)

        self.init_weight()

    def init_weight(self):
        trunc_normal_(self.cls_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, patches):
        patches = self.tokens(patches)
        patches = patches.view(patches.size(0), config.ADT_component, config.emb_ADT)
        patches = self.embedding(patches)
        patches = patches + self.pos_embedding

        patches = rearrange(patches, 'b c s -> c b s')
        patches, forward_indexes, backward_indexes = self.shuffle(patches)
        patches = torch.cat([self.cls_token.expand(-1,patches.shape[1],-1), patches], dim=0) 

        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.mamba(patches))
        features = rearrange(features, 'b t c -> t b c')

        return features, backward_indexes

# RNA decoder
class RNA_Decoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_RNA=10,RNA_component=400,decoder_layer=2
                 )-> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((RNA_component + 1, 1, emb_dim)))
        self.mamba = torch.nn.Sequential(*[Mamba(d_model=emb_dim) for _ in range(decoder_layer)])
        self.decoding = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_RNA))
        self.init_weight()
        
    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]
        
        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)

        features = features + self.pos_embedding
        features = rearrange(features, 't b c -> b t c')
        features = self.mamba(features)
        features = rearrange(features, 'b t c -> t b c')

        all_cls = []
        first_element = features[0]
        all_cls.append(first_element)
        
        patches = features[1:] 
        patches = self.decoding(patches)
        
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1

        mask = take_indexes(mask, backward_indexes[1:] - 1)
        patches = rearrange(patches, 't b c -> b t c')
        mask = rearrange(mask, 't b c -> b t c')
        patches = patches.reshape(patches.size(0),1,-1)
        mask = mask.reshape(mask.size(0),1,-1)

        return patches, mask, all_cls

# ADT_Decoder
class ADT_Decoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_ADT=10,ADT_component=11,decoder_layer=2
                 )-> None:
        super().__init__()

        self.mask_token = torch.nn.Parameter(torch.zeros(1, 1, emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((ADT_component + 1, 1, emb_dim)))##
        self.mamba = torch.nn.Sequential(*[Mamba(d_model=emb_dim) for _ in range(decoder_layer)])
        self.decoding = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_ADT))
        self.init_weight()
        
    def init_weight(self):
        trunc_normal_(self.mask_token, std=.02)
        trunc_normal_(self.pos_embedding, std=.02)

    def forward(self, features, backward_indexes):
        T = features.shape[0]

        backward_indexes = torch.cat([torch.zeros(1, backward_indexes.shape[1]).to(backward_indexes), backward_indexes + 1], dim=0)
        features = torch.cat([features, self.mask_token.expand(backward_indexes.shape[0] - features.shape[0], features.shape[1], -1)], dim=0)
        features = take_indexes(features, backward_indexes)

        features = features + self.pos_embedding
        features = rearrange(features, 't b c -> b t c')
        features = self.mamba(features)
        features = rearrange(features, 'b t c -> t b c')

        all_cls = []
        first_element = features[0]
        all_cls.append(first_element)
        
        patches = features[1:] 
        patches = self.decoding(patches)
       
        mask = torch.zeros_like(patches)
        mask[T-1:] = 1

        mask = take_indexes(mask, backward_indexes[1:] - 1)
        patches = rearrange(patches, 't b c -> b t c')
        mask = rearrange(mask, 't b c -> b t c')
        patches = patches.reshape(patches.size(0),1,-1)
        mask = mask.reshape(mask.size(0),1,-1)
        
        return patches, mask, all_cls
        
class scFuMamba_stage1(nn.Module):
    def __init__(self, config):
        super(scFuMamba_stage1, self).__init__()

        self.RNA_Encoder = RNA_Encoder(
            config.emb_dim, config.emb_RNA, config.RNA_component, 
            config.RNA_tokens,  config.encoder_layer, config.mask_ratio
        )
        self.ADT_Encoder = ADT_Encoder(
            config.emb_dim, config.emb_ADT, config.ADT_component, 
            config.ADT_tokens,  config.encoder_layer, config.mask_ratio1
        )      

        self.RNA_Decoder = RNA_Decoder(config.emb_dim, config.emb_RNA, config.RNA_component, config.decoder_layer)
        self.ADT_Decoder = ADT_Decoder(config.emb_dim, config.emb_ADT, config.ADT_component, config.decoder_layer)
        
        self.layer_norm1 = nn.LayerNorm(config.emb_dim)
        self.layer_norm2 = nn.LayerNorm(config.emb_dim)
        self.linear1 = nn.Linear(config.emb_dim, config.emb_dim)
        self.linear2 = nn.Linear(config.emb_dim, config.emb_dim)
        self.eca_layer = ECALayer(channel=config.emb_dim, k_size=config.k_size)
        self.mamba = Mamba(config.emb_dim)
        self.global_avg_pool = GlobalAveragePooling1d()
        self.ldc_rna = LDC(config.emb_dim, config.emb_dim)
        self.ldc_adt = LDC(config.emb_dim, config.emb_dim)
        self.max_pool_adjust = MaxPoolAdjust()

    def forward(self, patches1, patches2):
        omics_encoder_feature1, backward_indexes1 = self.RNA_Encoder(patches1)
        omics_encoder_feature2, backward_indexes2 = self.ADT_Encoder(patches2)

        omics_feature1 = rearrange(omics_encoder_feature1, 't b c -> b t c')
        omics_feature2 = rearrange(omics_encoder_feature2, 't b c -> b t c')

        omics_feature1_pooled = self.max_pool_adjust(omics_feature1, omics_feature2)

        _, omics_feature1_pooled_t, _ = omics_feature1_pooled.shape
        omics_feature1_cls = omics_feature1_pooled[:, omics_feature1_pooled_t-1, :].unsqueeze(1)
        omics_feature2_cls = omics_feature2[:, omics_feature1_pooled_t-1, :].unsqueeze(1)
        omics_feature_cross_cls = omics_feature1_cls + omics_feature2_cls

        avg_pool1 = self.global_avg_pool(omics_feature1_cls - omics_feature2_cls)
        avg_pool2 = self.global_avg_pool(omics_feature2_cls - omics_feature1_cls)
        avg_pool1 = F.sigmoid(avg_pool1)
        avg_pool2 = F.sigmoid(avg_pool2)

        omics_feature1_cross_cls = (
            avg_pool1 * omics_feature_cross_cls +
            self.ldc_rna(omics_feature1_cls) +
            omics_feature1_cls
        )

        omics_feature2_cross_cls = (
            avg_pool2 * omics_feature_cross_cls +
            self.ldc_adt(omics_feature2_cls) +
            omics_feature2_cls
        )

        omics_feature1_cls = self.linear1(self.layer_norm1(omics_feature1_cross_cls))
        omics_feature2_cls = self.linear2(self.layer_norm2(omics_feature2_cross_cls))

        omics_feature_cross = self.mamba(
            omics_feature1_cls * omics_feature2_cls +
            omics_feature1_cls + omics_feature2_cls
        )

        x1 = self.eca_layer(self.linear1(omics_feature_cross * omics_feature1_cls))
        x2 = self.eca_layer(self.linear2(omics_feature_cross * omics_feature2_cls))


        omics_feature1 = omics_feature1.clone()
        omics_feature2 = omics_feature2.clone()
        omics_feature1[:, omics_feature1_pooled_t-1, :] = x1.clone().squeeze(1)
        omics_feature2[:, omics_feature1_pooled_t-1, :] = x2.clone().squeeze(1)

        omics_encoder_feature1 = omics_encoder_feature1.clone()
        omics_encoder_feature2 = omics_encoder_feature2.clone()

        omics_feature1 = rearrange(omics_feature1.clone(), 'b t c -> t b c')
        omics_feature2 = rearrange(omics_feature2.clone(), 'b t c -> t b c')
        
        omics_feature1 = omics_feature1 + omics_encoder_feature1
        omics_feature2 = omics_feature2 + omics_encoder_feature2

        omics_patches1, mask1, all_cls1 = self.RNA_Decoder(omics_feature1, backward_indexes1) 
        omics_patches2, mask2, all_cls2 = self.ADT_Decoder(omics_feature2, backward_indexes2)
       
        return omics_patches1, omics_patches2, mask1, mask2, all_cls1, all_cls2





# Loading Dataset
RNA = torch.load('../dataset/CITE-seq/malt_10k_rna_rpkm.pth')
ADT = torch.load('../dataset/CITE-seq/malt_10k_prot_clred.pth')

from dataloader import *
# RNA
train_dataset,val_dataset = train_test_split(RNA,test_size=0.25, random_state=42)
train_dataset = train_dataset.to(torch.float).to(config.device)
val_dataset = val_dataset.to(torch.float).to(config.device)
M_train = len(train_dataset)
M_val = len(val_dataset)
# ADT
train_dataset1,val_dataset1 = train_test_split(ADT, test_size=0.25, random_state=42)
train_dataset1 = train_dataset1.to(torch.float).to(config.device)
val_dataset1 = val_dataset1.to(torch.float).to(config.device)
M_train1 = len(train_dataset1)
M_val1 = len(val_dataset1)

multi_modal_trian_dataset = MultiModalDataset(train_dataset, train_dataset1)
multi_modal_test_dataset = MultiModalDataset(val_dataset, val_dataset1)
train_dataloader = torch.utils.data.DataLoader(multi_modal_trian_dataset, 128, shuffle=True,num_workers=0)
val_dataloader = torch.utils.data.DataLoader(multi_modal_test_dataset, 128, shuffle=False,num_workers=0)

# training
early_stopping_patience = 5  
best_val_loss = float('inf')  
no_improvement_count = 0
weight_a = 0.7
weight_b = 0.3

model = scFuMamba_stage1(config).to(config.device)

if __name__ == '__main__':
    
        batch_size = config.batch_size
        load_batch_size = 128
        assert batch_size % load_batch_size == 0
        steps_per_update = batch_size // load_batch_size

        optim = torch.optim.AdamW(model.parameters(), lr=config.lr * config.batch_size / 256, betas=(0.9, 0.999), weight_decay=1e-4)
        lr_func = lambda epoch: min((epoch + 1) / (config.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / config.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

        best_val_acc = 0
        step_count = 0
        optim.zero_grad()
        
        train_losses_list = []
        val_losses_list = []
        for e in range(config.total_epoch):
            model.train()
            train_losses = []
            for tk in tqdm(iter(train_dataloader)):
               
                step_count += 1

                
                RNA_patches,ADT_patches, mask1,mask2,all_RNAcls, all_ADTcls = model(tk['mod1'], tk['mod2'])
                
                loss_a = torch.mean((RNA_patches - tk['mod1']) ** 2 * mask1) / config.mask_ratio
                loss_b = torch.mean((ADT_patches - tk['mod2']) ** 2 * mask2) / config.mask_ratio1
                train_loss = weight_a * loss_a + weight_b * loss_b
                train_loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                train_losses.append(train_loss.item())
            lr_scheduler.step()
            avg_train_loss = sum(train_losses) / len(train_losses)
            train_losses_list.append(avg_train_loss)
            print(f'In epoch {e}, average training loss is {avg_train_loss}.')

            model.eval()
            with torch.no_grad():
                val_losses = []
                for td in tqdm(iter(val_dataloader)):
                    RNA_patches_val,ADT_patches_val, mask1_val,mask2_val, all_RNAcls_val, all_ADTcls_val = model(td['mod1'], td['mod2'])
                    
                    loss_c = torch.mean((RNA_patches_val - td['mod1']) ** 2 * mask1_val) / config.mask_ratio
                    loss_d = torch.mean((ADT_patches_val - td['mod2']) ** 2 * mask2_val) / config.mask_ratio1           
                    val_loss = weight_a * loss_c + weight_b * loss_d
                    val_losses.append(val_loss.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                val_losses_list.append(avg_val_loss)
                print(f'In epoch {e}, average validation loss is {avg_val_loss}.')  

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0  #
                print(f'Saving best model with loss {best_val_loss} at epoch {e}!')

                #save best paras
                # torch.save(model.state_dict(), '/path/to/save/your_model.pth')
 
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f'No improvement in validation loss for {early_stopping_patience} epochs. Early stopping!')
                break  