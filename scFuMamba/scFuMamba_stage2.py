import torch
import torch.nn as nn
import timm
import time
import numpy as np
import os
import math
from einops import repeat, rearrange
from timm.models.layers import trunc_normal_
import anndata as ad
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn.parallel import DataParallel
import pandas as pd
from torch.utils.data import DataLoader
import warnings
from mamba_ssm import Mamba
import torch.optim as optim
import torch.nn.functional as F
from utils import *
warnings.filterwarnings("ignore")

class Config(object):
    """para"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
        self.dropout = 0.1                                    
        self.num_classes = 18                                            
        self.batch_size = 128                                       
        self.lr = 5e-4                      
        self.encoder_layer = 1
        self.mask_ratio = 0.15
        
        self.RNA_tokens = 4000 ##RNA number
        self.RNA_component = 2000 ##channel        
        self.emb_RNA = 2
    
        self.mask_ratio1 = 0.15
        
        self.ADT_tokens = 14 ##proteins umber
        self.ADT_component = 14 ##channel
        self.emb_ADT = 1 
        
        self.emb_dim = 128       
        self.total_epoch = 500
        self.warmup_epoch = 10
        self.k_size = 1

config = Config()        


# Starting Stage2
class RNA_Encoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_RNA=10,RNA_component=400,RNA_tokens=4000,encoder_layer=6) -> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = RNA_tokens, out_features = RNA_tokens))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_RNA, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, RNA_component ,emb_dim)))##
        self.mamba = nn.Sequential(*[Mamba(d_model=emb_dim) for _ in range(encoder_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)


    def forward(self, patches):
        patches = self.tokens(patches)
        patches = patches.view(patches.size(0),config.RNA_component,config.emb_RNA)
        patches = self.embedding(patches)
        patches = patches + self.pos_embedding
        patches = rearrange(patches, 'b c s -> c b s')
        
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.mamba(patches))  
        features = rearrange(features, 'b t c -> t b c')

        
        return features

class ADT_Encoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_ADT=10,ADT_component=400,ADT_tokens=4000,encoder_layer=6) -> None:
        super().__init__()
        self.tokens = torch.nn.Sequential(torch.nn.Linear(in_features = ADT_tokens, out_features = ADT_tokens))
        self.embedding = torch.nn.Sequential(torch.nn.Linear(in_features = emb_ADT, out_features = emb_dim))
        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1,emb_dim))
        self.pos_embedding = torch.nn.Parameter(torch.zeros((1, ADT_component ,emb_dim)))#
        self.mamba = nn.Sequential(*[Mamba(d_model=emb_dim) for _ in range(encoder_layer)])
        self.layer_norm = torch.nn.LayerNorm(emb_dim)


    def forward(self, patches):
        patches = self.tokens(patches)
        patches = patches.view(patches.size(0),config.ADT_component,config.emb_ADT)
        patches = self.embedding(patches)
        patches = patches + self.pos_embedding
        patches = rearrange(patches, 'b c s -> c b s')
        
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')

        features = self.layer_norm(self.mamba(patches)) 
        features = rearrange(features, 'b t c -> t b c')

        
        return features
    
class scFuMamba_stage2(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.RNA_Encoder = RNA_Encoder(config.emb_dim,config.emb_RNA,config.RNA_component,config.RNA_tokens,config.encoder_layer)
        self.ADT_Encoder = ADT_Encoder(config.emb_dim,config.emb_ADT,config.ADT_component,config.ADT_tokens,config.encoder_layer)

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
        self.head = nn.Linear(config.emb_dim, config.num_classes)

    def forward(self, patches1,patches2):
      
        patches1 = self.RNA_Encoder(patches1)
        patches2 = self.ADT_Encoder(patches2)

        omics_feature1 = rearrange(patches1.clone(), 't b c -> b t c')
        omics_feature2 = rearrange(patches2.clone(), 't b c -> b t c')

        omics_feature1_pooled = self.max_pool_adjust(omics_feature1, omics_feature2)
        _, omics_feature1_pooled_t, _ = omics_feature1_pooled.shape
        omics_feature1_cls = omics_feature1_pooled[:, (omics_feature1_pooled_t-1), :].unsqueeze(1)
        omics_feature2_cls = omics_feature2[:, (omics_feature1_pooled_t-1), :].unsqueeze(1)
        omics_feature_cross_cls = (omics_feature1_cls + omics_feature2_cls)
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

        omics_feature_cross = (
            omics_feature1_cls * omics_feature2_cls +
            omics_feature1_cls + omics_feature2_cls
        )
        rna_cls = omics_feature1_cls
        prot_cls = omics_feature2_cls
        
        x1 = self.eca_layer(self.linear1(omics_feature_cross * omics_feature1_cls))
        x2 = self.eca_layer(self.linear2(omics_feature_cross * omics_feature2_cls))
        x1 = x1 + omics_feature1_cls + omics_feature_cross
        x2 = x2 + omics_feature2_cls + omics_feature_cross 
        
        
        final_results = []
        final_result = (x1+x2)/2
        final_results.append(final_result.squeeze(1))
        logits = self.head(final_result)
        logits = logits.squeeze(1)
        return logits,final_results,rna_cls,prot_cls
    
    
# paired omics with label  Dataset
from dataloaderwithlabel import *
from dataloader import MultiModalDataset
omic1 = torch.load('../dataset/CITE-seq/malt_10k_rna_rpkm.pth')
omic2 = torch.load('../dataset/CITE-seq/malt_10k_prot_clred.pth')
labels = np.load('../dataset/CITE-seq/malt10k_6838wnn_labels.npy',allow_pickle=True)
labels = labels.astype(int)

# Dataloader
# RNA
train_dataset,val_dataset,y_train,y_test = train_test_split(omic1,labels,test_size=0.7, random_state=42)
# ADT
train_dataset1,val_dataset1,y_train1,y_test1 = train_test_split(omic2,labels,test_size=0.7, random_state=42)

# Second split,just only use 30% dataset to finetune
train_dataset,val_dataset,y_train,y_test = train_test_split(train_dataset,y_train,test_size=0.1, random_state=42)
train_dataset = train_dataset.to(torch.float).to(config.device)
val_dataset = val_dataset.to(torch.float).to(config.device)
y_train = torch.tensor(y_train, dtype=torch.long).to(config.device)
y_test = torch.tensor(y_test, dtype=torch.long).to(config.device)
# ADT
train_dataset1, val_dataset1, y_train1, y_test1 = train_test_split(train_dataset1,y_train1,test_size=0.1, random_state=42)
train_dataset1 = train_dataset1.to(torch.float).to(config.device)
val_dataset1 = val_dataset1.to(torch.float).to(config.device)

multi_modal_trian_dataset = MultiModalDataset_label(train_dataset, train_dataset1,y_train)
multi_modal_test_dataset = MultiModalDataset_label(val_dataset, val_dataset1,y_test)

train_dataloader = torch.utils.data.DataLoader(multi_modal_trian_dataset, 128, shuffle=True,num_workers=0)
val_dataloader = torch.utils.data.DataLoader(multi_modal_test_dataset, 128, shuffle=False,num_workers=0)

train_dataset.shape,train_dataset1.shape,val_dataset.shape,val_dataset1.shape,y_train.shape,y_test.shape



# loading pretrained stage1-model
model = scFuMamba_stage2(config).to(config.device)
# Load the state dictionary
# model.load_state_dict(torch.load(f'{Your Path}/scFuMamba_{dataset}_pretrain_{e}epoch_best_model.pth'),strict=False)

# Training
early_stopping_patience = 5  
best_val_loss = float('inf')  
no_improvement_count = 0
# loss
lossfun = torch.nn.CrossEntropyLoss()
acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

model = scFuMamba_stage2(config).to(config.device)
if __name__ == '__main__':
    
        batch_size = config.batch_size
        load_batch_size = 128
        assert batch_size % load_batch_size == 0
        steps_per_update = batch_size // load_batch_size

        optim = torch.optim.Adam(model.parameters(), lr=config.lr * config.batch_size / 256, betas=(0.9, 0.999), weight_decay=3e-2)
        lr_func = lambda epoch: min((epoch + 1) / (config.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / config.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

        best_val_acc = 0
        step_count = 0
        optim.zero_grad()
        train_loss_list = []
        train_acc_list  = []
        val_loss_list = []
        val_acc_list  = []           
        for e in range(config.total_epoch):
            model.train()
            train_losses = []
            train_acces  = []
            for tk in tqdm(iter(train_dataloader)):
                step_count += 1
                
                train_logits,final_results,rna_cls,prot_cls = model(tk['mod1'], tk['mod2'])
                train_loss = lossfun(train_logits,tk['label'])
                train_acc = acc_fn(train_logits,tk['label'])
                train_loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()
                train_losses.append(train_loss.item())
                train_acces.append(train_acc.item())
            lr_scheduler.step()
            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_acc = sum(train_acces) / len(train_acces)
            train_loss_list.append(avg_train_loss)
            train_acc_list.append(avg_train_acc)
            print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')

            model.eval()
            with torch.no_grad():
                val_losses = []
                val_acces = []
                for td in tqdm(iter(val_dataloader)):
                    val_logits,final_results2,rna_cls2,prot_cls2 = model(td['mod1'], td['mod2'])
                    val_loss = lossfun(val_logits,td['label'])
                    val_acc = acc_fn(val_logits,td['label'])
                    val_losses.append(val_loss.item())
                    val_acces.append(val_acc.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_acc = sum(val_acces) / len(val_acces)
                val_loss_list.append(avg_val_loss)
                val_acc_list.append(avg_val_acc)
                print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0  #
                print(f'Saving best model with loss {best_val_loss} at epoch {e}!')

                # save best paras
                # torch.save(model.state_dict(), '/path/to/save/your_model.pth') 
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f'No improvement in validation loss for {early_stopping_patience} epochs. Early stopping!')
                break 
                