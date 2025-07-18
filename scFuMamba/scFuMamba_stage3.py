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
from dataloaderwithlabel import *
from dataloader import MultiModalDataset
from utils import *
warnings.filterwarnings("ignore")

class Config(object):
    """para"""
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')  
        self.dropout = 0.1                                 
        self.num_classes = 13                                        
        self.batch_size = 128                   
        self.lr = 5e-4
        self.encoder_layer = 1
        self.encoder_head = 2
        self.mask_ratio = 0.15
        
        self.RNA_tokens = 4000 ##RNA number
        self.RNA_component = 2000 ##channel        
        self.emb_RNA = 2
    
        self.mask_ratio1 = 0.15
        
        self.emb_dim = 128      
        self.total_epoch = 500
        self.warmup_epoch = 10
        self.k_size = 1

config = Config()        

###Starting Stage2
class RNA_Encoder(torch.nn.Module):
    def __init__(self,emb_dim=64,emb_RNA=10,RNA_component=400,RNA_tokens=4000, encoder_head=4,encoder_layer=6) -> None:
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


class scFuMamba_stage3(torch.nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.RNA_Encoder = RNA_Encoder(config.emb_dim,config.emb_RNA,config.RNA_component,config.RNA_tokens,config.encoder_head,config.encoder_layer)
        self.layer_norm1 = nn.LayerNorm(config.emb_dim)
        self.layer_norm2 = nn.LayerNorm(config.emb_dim)
        self.linear1 = nn.Linear(config.emb_dim, config.emb_dim)
        self.linear2 = nn.Linear(config.emb_dim, config.emb_dim)
        self.eca_layer = ECALayer(channel=config.emb_dim, k_size=config.k_size)
        self.mamba = Mamba(config.emb_dim)
        self.global_avg_pool = GlobalAveragePooling1d()
        self.MinPooling = MinPooling1d(config.k_size,stride=1,padding=0)
        self.ldc_rna = LDC(config.emb_dim, config.emb_dim)
        self.ldc_adt = LDC(config.emb_dim, config.emb_dim)
        self.head = nn.Linear(config.emb_dim, config.num_classes)
    def forward(self, patches1):

        patches1 = self.RNA_Encoder(patches1)
  
        omics_feature1 = rearrange(patches1.clone(), 't b c -> b t c')
        omics_feature2 = rearrange(patches1.clone(), 't b c -> b t c')
        _, omics_feature1_t, _ = omics_feature1.shape
        omics_feature1_cls = omics_feature1[:, omics_feature1_t-1, :].unsqueeze(1)
        omics_feature2_cls = omics_feature2[:, omics_feature1_t-1, :].unsqueeze(1)
        omics_feature_cross_cls = self.mamba(omics_feature1_cls + omics_feature2_cls)
        avg_pool1 = self.global_avg_pool(omics_feature1_cls - self.MinPooling(omics_feature2_cls))
        avg_pool1 = F.sigmoid(avg_pool1)
        omics_feature1_cross_cls = self.mamba((
            avg_pool1 * omics_feature_cross_cls +
            self.ldc_rna(omics_feature1_cls) +
            omics_feature1_cls
        ))
        omics_feature1_cross_cls = omics_feature1_cls + omics_feature1_cross_cls
        
        final_results = []
        final_result = omics_feature1_cross_cls
        final_results.append(final_result.squeeze(1))##
        
        logits = self.head(final_result)
        logits = logits.squeeze(1)

        return logits,final_results



from dataloaderwithlabel import *
omic1 = torch.load('../dataset/RNA-seq/ifnb_rna_rpkm_normalized.pth')
omic1 = omic1.float()
labels = pd.read_csv('../dataset/RNA-seq/ifnb_label.csv')
labels = labels['x']
labels = np.array(labels)

config = Config()
##RNA
train_dataset,val_dataset,y_train,y_test = train_test_split(omic1,labels,test_size=0.7, random_state=42,shuffle=False)
###second split,just 30 percent data to finetune
train_dataset,val_dataset,y_train,y_test = train_test_split(train_dataset,y_train,test_size=0.1, random_state=42)
train_dataset = train_dataset.to(torch.float).to(config.device)
val_dataset = val_dataset.to(torch.float).to(config.device)
y_train = torch.tensor(y_train, dtype=torch.long).to(config.device)
y_test = torch.tensor(y_test, dtype=torch.long).to(config.device)

multi_modal_trian_dataset = SingleModalDataset(train_dataset,y_train)
multi_modal_test_dataset = SingleModalDataset(val_dataset,y_test)

train_dataloader = torch.utils.data.DataLoader(multi_modal_trian_dataset, 128, shuffle=False,num_workers=0)
val_dataloader = torch.utils.data.DataLoader(multi_modal_test_dataset, 128, shuffle=False,num_workers=0)

# loading pretrained stage2-model
#config = Config()
#model = scFuMamba_stage3(config).to(config.device)
#model.load_state_dict(torch.load(f'{Your Path}/{Dataset}_finetune_best_model.pth'),strict=False)


# Training
early_stopping_patience = 5  
best_val_loss = float('inf')  
no_improvement_count = 0
# loss
lossfun = torch.nn.CrossEntropyLoss()
acc_fn = lambda logit, label: torch.mean((logit.argmax(dim=-1) == label).float())

model = scFuMamba_stage3(config).to(config.device)
model = DataParallel(model)
if __name__ == '__main__':
    
        batch_size = config.batch_size
        load_batch_size = 128
        assert batch_size % load_batch_size == 0
        steps_per_update = batch_size // load_batch_size

        optim = torch.optim.AdamW(model.parameters(), lr=config.lr * config.batch_size / 256, betas=(0.9, 0.999), weight_decay=4e-2)
        lr_func = lambda epoch: min((epoch + 1) / (config.warmup_epoch + 1e-8), 0.5 * (math.cos(epoch / config.total_epoch * math.pi) + 1))
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_func, verbose=True)

        best_val_acc = 0
        step_count = 0
        optim.zero_grad()

        for e in range(config.total_epoch):
            model.train()
            train_losses = []
            train_acces = []

            for tk in tqdm(iter(train_dataloader)):
                step_count += 1

                train_logits, final_results = model(tk['mod1'])
                train_loss = lossfun(train_logits, tk['label'])
                train_acc = acc_fn(train_logits, tk['label'])

                train_loss.backward()
                if step_count % steps_per_update == 0:
                    optim.step()
                    optim.zero_grad()

                train_losses.append(train_loss.item())
                train_acces.append(train_acc.item())

            lr_scheduler.step()

            avg_train_loss = sum(train_losses) / len(train_losses)
            avg_train_acc = sum(train_acces) / len(train_acces)

            print(f'In epoch {e}, average training loss is {avg_train_loss}, average training acc is {avg_train_acc}.')


            model.eval()
            with torch.no_grad():
                val_losses = []
                val_acces = []
                for td in tqdm(iter(val_dataloader)):
                    val_logits,final_results2= model(td['mod1'])
                    
                    val_loss = lossfun(val_logits,td['label'])
                    val_acc = acc_fn(val_logits,td['label'])
                    val_losses.append(val_loss.item())
                    val_acces.append(val_acc.item())
                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_acc = sum(val_acces) / len(val_acces)
                print(f'In epoch {e}, average validation loss is {avg_val_loss}, average validation acc is {avg_val_acc}.')  

        # 
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                no_improvement_count = 0  #
                print(f'Saving best model with loss {best_val_loss} at epoch {e}!')

                # save best paras
                #torch.save(model.state_dict(), '/path/to/save/your_model.pth')
            else:
                no_improvement_count += 1

            if no_improvement_count >= early_stopping_patience:
                print(f'No improvement in validation loss for {early_stopping_patience} epochs. Early stopping!')
                break  #  
        

# Predict
MultiModal_dataset = MultiModalDataset(omic1,omic1)
all_dataloader = torch.utils.data.DataLoader(MultiModal_dataset, 128, shuffle=False,num_workers=0)

model.eval() 
fin = []
with torch.no_grad():
    for batch in all_dataloader:
        inputs1 = batch['mod1'].to(config.device)
        inputs2 = batch['mod2'].to(config.device)
        try:
            final_logits,final_results = model(inputs1)
            fin.append(final_results[0].cpu().numpy())
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Not enough video memory, try reducing the batch size")
            else:
                raise e

fin = np.concatenate(fin, axis=0)
print(fin.shape)


# UMAP results and calculatin metrics
from umappre import *
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.manifold import TSNE
import pandas as pd
import secuer as sr

# community detection
adj_matrix = knn_adj_matrix(fin)
y_pred = RunLeiden(adj_matrix)# Community detection
# UMAP reduction
reducer = umap.UMAP()
embedding = reducer.fit_transform(fin)

df = pd.DataFrame({'x': pd.Series(embedding[:, 0]), 'y': pd.Series(embedding[:, 1]), 'label': y_pred})
plt.figure(figsize=(8, 6))
sns.scatterplot(data=df, x='x', y='y', hue='label', s=10,palette = sns.color_palette(mpl.colors.TABLEAU_COLORS ,len(set(y_pred))),)
plt.title('UMAP projection of the 151673', fontsize=20)
plt.legend(bbox_to_anchor=(1.05, -0), loc=3, borderaxespad=4)
plt.savefig(os.path.join("Your_path", "UMAP.png"))
print("UMAP Done!")


###calculating
y_test = labels
print("unique_labels:",np.array(set(y_pred)))
from metric import *
from sklearn import metrics
print("Embedding shape:", embedding.shape)
print("y_test shape:", y_test.shape)

metrics_dict = {
    'Mean Average Precision': mean_average_precision(embedding, np.ravel(y_test)),
    'Avg Silhouette Width': avg_silhouette_width(embedding, np.ravel(y_test)),
    'Graph Connectivity': graph_connectivity(embedding, np.ravel(y_test)),
    'ARI': metrics.adjusted_rand_score(np.ravel(y_test), np.ravel(y_pred)),
    'NMI': metrics.normalized_mutual_info_score(np.ravel(y_test), np.ravel(y_pred)),
    'FMI': metrics.fowlkes_mallows_score(np.ravel(y_test), np.ravel(y_pred)),
    'Silhouette Coefficient': metrics.silhouette_score(embedding, y_pred, metric='euclidean'),
    'Calinski-Harabaz Index': metrics.calinski_harabasz_score(embedding, y_pred),
    'Davies-Bouldin Index': metrics.davies_bouldin_score(embedding, y_pred),
    'Purity': purity(y_pred, y_test),
    'AMI': metrics.adjusted_mutual_info_score(y_test, y_pred),
    'Homogeneity': metrics.homogeneity_score(y_test, y_pred),
    'Completeness': metrics.completeness_score(y_test, y_pred),
    'V-measure': metrics.v_measure_score(y_test, y_pred),
    'F-measure': F_measure(y_pred, y_test),
    'Jaccard Index': jaccard(y_pred, y_test),
    'Dice Index': Dice(y_pred, y_test)
}

df = pd.DataFrame(list(metrics_dict.items()), columns=['Metric', 'scMMAE_cbmc'])
df.to_csv('Your_path', index=False)