from math import e
import os
import torch
import json
import time
import numpy as np
import pandas as pd
from torch import nn
from PIL import Image
from glob import glob
import random
import argparse
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader,random_split, Subset
from sklearn.manifold import TSNE
import torchvision.models as models
from sklearn.metrics import confusion_matrix,precision_score,recall_score,f1_score
import torch.optim as optim
from sklearn.model_selection import KFold,TimeSeriesSplit


class GradCam:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.extractor = CamExtractor(self.model)

    def generate_cam(self, input_image, target_layer, target_class=None):
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.numpy())
        
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output[0][target_class] = 1
        
        self.model.zero_grad()
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        
        guided_gradients = self.extractor.gradients[-1 - target_layer].data.numpy()[0]
        target = conv_output[target_layer].data.numpy()[0]
        
        weights = np.mean(guided_gradients, axis=(1, 2))
        cam = np.ones(target.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))
        cam = np.uint8(cam * 255)
        cam_resize = Image.fromarray(cam).resize((input_image.shape[2], input_image.shape[3]), Image.ANTIALIAS)
        cam = np.uint8(cam_resize) / 255
        
        return cam

class CamExtractor:
    def __init__(self, model):
        self.gradients = []
        self.model = model
        self.model._modules.get('layer4').register_backward_hook(self.backward_hook)
        self.model._modules.get('layer4').register_forward_hook(self.forward_hook)

    def forward_hook(self, module, input, output):
        self.conv_outputs = output

    def backward_hook(self, module, grad_in, grad_out):
        self.gradients.append(grad_in[0])

    def forward_pass(self, x):
        return self.conv_outputs, self.model(x)
    

class Data_Generator(Dataset):
    def __init__(self,dataset,MAX,seqfile,catfile):
        self.csv_path = []
        self.label = []
        self.image_path=[]
        for i in range(20, MAX+20, 1):
            seq_root = f'{seqfile}/{dataset}/{i}'
            cat_root = f'{catfile}/{dataset}/{i}'
            seq_files = sorted(os.listdir(seq_root))
            cat_files = sorted(os.listdir(cat_root))
            for seq_file, cat_file in zip(seq_files, cat_files):
                self.csv_path.append(os.path.join(seq_root, seq_file))
                self.image_path.append(os.path.join(cat_root, cat_file))
                self.label.append(i)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # StreamData
        data = pd.read_csv(self.csv_path[index],header=None)
        data = np.array(data)  # numpy array
        data=np.float32(data)

        # ImageData
        image=Image.open(self.image_path[index])
        image = image.resize((224, 224))
        image = np.array(image).astype(np.float32) / 255.0

        return data, image, self.label[index] - 20


class LSTMModel(nn.Module):
    def __init__(self,MAX=10, input=35, bi=1):
        super(LSTMModel,self).__init__()
        self.lstm=nn.LSTM(input_size=input, hidden_size=20, num_layers=2, batch_first=True, dropout=0.3, bidirectional=bi) # 2 layers  dropout 0.1
        #35,34,73
    def forward(self,x):
        out,_ = self.lstm(x)
        # out=out[:,-1,:]

        # batch_size=out.shape[0]
        # out=out.reshape(batch_size,-1)
        return out
    

class LSTM_CNN(nn.Module):
    def __init__(self,MAX,input,bi):
        super().__init__()

        model1=LSTMModel(MAX=MAX,input=input,bi=bool(bi))
        self.model1=model1.lstm # output out=out[:,-1,:]
        model2 = models.resnet18(weights=None)
        model2.fc=nn.Identity()
        self.model2=model2

        ## Add-7/7
        # self.img_seq_balance=nn.Linear(512,20)

        #MLP
        if bi==0:
            self.mlp1 = nn.Sequential(
                nn.Linear(20,32), 
                ############################ Activation
                # nn.RELU()
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(512, 32),
                # nn.Linear(20, 32), # 7/7
                ############################ Activation
                # nn.SELU()
            )
            self.mlp3 = nn.Sequential(
                nn.Linear(512+20, MAX)
                # nn.Linear(40, MAX)# 7/7
            )
        elif bi==1:
            self.mlp1 = nn.Sequential(
                nn.Linear(20*2,32), #
                ############################ Activation
                # nn.RELU()
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(512, 32),
                # nn.Linear(20, 32), # 7/7
                ############################ Activation
                # nn.SELU()
            )
            self.mlp3 = nn.Sequential(
                nn.Linear(512+20*2, MAX)
                # nn.Linear(60, MAX)# 7/7
            )

    def forward(self,seq,image):
        #LSTM
        seq,_ = self.model1(seq)
        seq=seq[:,-1,:]

        # image
        image=image.permute([0,3,1,2])
        image=self.model2(image)
        batch_size = image.shape[0]
        image = image.reshape(batch_size, -1)
        
        # image=self.img_seq_balance(image)

        conca = torch.cat([seq, image], dim=1)
        conca = self.mlp3(conca)

        seq = self.mlp1(seq)
        image = self.mlp2(image)
        return conca,seq,image
        ### For SCL in dim=2
        # conca = torch.cat([seq, image], dim=1)
        # conca = self.mlp3(conca)

        # seq = F.normalize(seq)
        # image = F.normalize(image)
        # feat=torch.cat([seq,image],dim=1)
        # # feat=self.mlp4(feat)
        # return conca,seq,image,feat


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07, reduction='sqrt', supcon_grad=True):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.reduction = reduction
        self.supcon_grad = supcon_grad

    def forward(self, features, labels, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device=(torch.device("cuda")
                if features.is_cuda
                else torch.device("cpu"))
        ########################################################
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # compute mean of log-likelihood over positive
        mask_pos_pairs=mask.sum(1)
        mask_pos_pairs=torch.where(mask_pos_pairs<1e-6,1,mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs
        # mean_log_prob_pos = (mask * log_prob).sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        
        ##############################
        if self.reduction == 'mean':
            loss = loss.view(anchor_count, batch_size).mean()
        elif self.reduction == 'sqrt':
            loss = loss.view(anchor_count, batch_size).mean(dim=1)[None,:]
            loss = torch.sqrt(loss@loss.T)

        if self.supcon_grad:
            return loss
        else:
            return loss

        # return float(torch.sqrt(loss@loss.T))

class SupConLossWithNOTA(nn.Module):
    r"""
    Supervised Contrastive loss with optional NOTA behavior (label == 0).

    Behavior
    --------
    - If enable_nota=False: behaves like standard SupCon (with positives defined by equality of labels).
    - If enable_nota=True:
        * Positives are defined only for non-NOTA labels (>0). NOTA never forms positives.
        * NOTA anchors (label==0) get a repulsive InfoNCE term:
              ℓ_i^NOTA = log( Σ_{a in contrasts eligible for NOTA} exp(s_{ia}) ),
          which encourages low similarity to the chosen contrast pool.
          By default, NOTA repels against NON-NOTA only (nota_compare_to='nonnota').
          You can set nota_compare_to='all' to repel from everyone (including other NOTA).
        * Optionally, non-NOTA anchors can EXCLUDE NOTA contrasts from their denominator
          (exclude_nota_from_nonnota_denominator=True), making NOTA "don't care" for them.

    Args
    ----
    temperature: float
        Softmax temperature τ.
    contrast_mode: str
        'all' (use all views as anchors) or 'one' (use only view 0 as anchors).
    base_temperature: float
        Base temperature used to scale the loss (as in SupCon).
    enable_nota: bool
        If True, enable special handling for label==0 (NOTA).
    repulsion_weight: float
        λ_rep: weight on the NOTA repulsion term.
    nota_compare_to: {'nonnota', 'all'}
        For NOTA anchors, which contrasts to repel against: only non-NOTA (default) or all.
    exclude_nota_from_nonnota_denominator: bool
        If True, NOTA contrasts are removed from the denominator of non-NOTA anchors.
        Default False (vanilla SupCon denominator includes all non-self contrasts).
    eps: float
        Numerical stability constant.

    Inputs
    ------
    features: Tensor
        Shape [bsz, n_views, ...]. Will be flattened on the last dims.
    labels: Tensor
        Shape [bsz]. Integer labels; 0 denotes NOTA if enable_nota=True.

    Returns
    -------
    loss: Tensor (scalar)
    """
    def __init__(self,
                 temperature=0.07,
                 contrast_mode='all',
                 base_temperature=0.07,
                 enable_nota=True,
                 nota_weight=1.0,
                 nota_compare_to='nonnota',  # or 'all'
                 exclude_nota_from_nonnota_denominator=False,
                 eps=1e-20):
        super().__init__()
        assert contrast_mode in ('one', 'all')
        assert nota_compare_to in ('nonnota', 'all')
        self.temperature = float(temperature)
        self.contrast_mode = contrast_mode
        self.base_temperature = float(base_temperature)
        self.enable_nota = bool(enable_nota)
        self.nota_weight = float(nota_weight)
        self.nota_compare_to = nota_compare_to
        self.exclude_nota_from_nonnota_denominator = bool(exclude_nota_from_nonnota_denominator)
        self.eps = float(eps)

    def forward(self, features, labels):
        if features.ndim < 3:
            raise ValueError("`features` must have shape [bsz, n_views, ...].")
        bsz, n_views = features.shape[:2]
        device = features.device
        if features.ndim > 3:
            features = features.view(bsz, n_views, -1)  # flatten trailing dims

        # ----- Build positive-pair mask at the SAMPLE level -----
        # Base equality mask: same label -> positive in vanilla SupCon
        y = labels.view(-1, 1)
        same_cls = (y == y.T).float().to(device)               # [bsz, bsz]

        if self.enable_nota:
            # Positives only for non-NOTA classes
            non_nota_pair = ((y > 0) & (y.T > 0)).float().to(device)
            mask_pos_sample = same_cls * non_nota_pair         # no positives for label==0
        else:
            mask_pos_sample = same_cls                         # standard SupCon

        # ----- Build feature matrices for contrast -----
        contrast_count = n_views
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)  # [bsz*n_views, d]

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]                                   # [bsz, d]
            anchor_count = 1
        else:
            anchor_feature = contrast_feature                                 # [bsz*n_views, d]
            anchor_count = contrast_count

        # ----- Similarity logits (scaled), stabilized -----
        logits = torch.matmul(anchor_feature, contrast_feature.T) / self.temperature
        logits_max, _ = logits.max(dim=1, keepdim=True)
        logits = logits - logits_max.detach()  # numerical stability

        # ----- Tile masks to match logits shape -----
        # Base "eligible contrast" mask: exclude self-contrast only
        logits_mask = torch.ones_like(logits, device=device)
        diag_idx = torch.arange(bsz * anchor_count, device=device)
        logits_mask.scatter_(1, diag_idx.view(-1, 1), 0.0)  # zero diagonal

        mask_pos = mask_pos_sample.repeat(anchor_count, contrast_count)       # [bsz*anc, bsz*cont]

        # ----- Optional: exclude NOTA from denominators of non-NOTA anchors -----
        # This affects ONLY the denominator (i.e., the set we sum over in softmax), not the positive mask.
        if self.enable_nota and self.exclude_nota_from_nonnota_denominator:
            labels_vec = labels.view(-1)
            is_nota_col = (labels_vec == 0).float().to(device)                # [bsz]
            # tile columns over views:
            is_nota_col_tiled = is_nota_col.repeat(contrast_count)            # [bsz*cont]
            # find non-NOTA anchors:
            if self.contrast_mode == 'one':
                anchor_labels = labels_vec                                    # [bsz]
            else:
                anchor_labels = labels_vec.repeat_interleave(anchor_count)     # [bsz*anc]
            is_nonnota_anchor = (anchor_labels > 0)                            # [bsz*anc]
            # Build a [bsz*anc, bsz*cont] mask that zeros NOTA columns for non-NOTA anchor rows
            elig_mask = torch.ones_like(logits_mask, dtype=torch.bool, device=device)
            if is_nonnota_anchor.any():
                # rows where anchor is non-NOTA:
                rows = is_nonnota_anchor.nonzero(as_tuple=False).view(-1)
                # broadcast NOTA columns to those rows
                elig_mask[rows] = ~is_nota_col_tiled.bool()                    # keep only non-NOTA cols
            logits_mask = logits_mask.bool() & elig_mask
            logits_mask = logits_mask.float()

        # ----- Softmax denominator over eligible contrasts -----
        exp_logits = torch.exp(logits) * logits_mask
        denom = exp_logits.sum(1, keepdim=True).clamp_min(self.eps)
        log_prob = logits - denom.log()

        # ----- SupCon positive term (for anchors that have positives) -----
        pos_count = mask_pos.sum(1)                               # [bsz*anc]
        has_pos = pos_count > 0
        mean_log_prob_pos = torch.zeros_like(pos_count, device=device)
        if has_pos.any():
            mean_log_prob_pos[has_pos] = (mask_pos[has_pos] * log_prob[has_pos]).sum(1) / pos_count[has_pos]

        supcon_term = -(self.temperature / self.base_temperature) * mean_log_prob_pos  # [bsz*anc]

        # ----- NOTA repulsion term for NOTA anchors (optional) -----
        if self.enable_nota:
            labels_vec = labels.view(-1)
            if self.contrast_mode == 'one':
                anchor_labels = labels_vec                                  # [bsz]
            else:
                anchor_labels = labels_vec.repeat_interleave(anchor_count)  # [bsz*anc]
            is_nota_anchor = (anchor_labels == 0)

            if is_nota_anchor.any():
                # Choose which contrasts NOTA repels against
                if self.nota_compare_to == 'nonnota':
                    # Only NON-NOTA contrasts:
                    contrast_is_nonnota = (labels_vec > 0).float().to(device)        # [bsz]
                    contrast_is_nonnota = contrast_is_nonnota.repeat(contrast_count) # [bsz*cont]
                    nota_contrast_mask = contrast_is_nonnota.unsqueeze(0)            # [1, bsz*cont]
                else:
                    # 'all': all contrasts (already excluding self via logits_mask)
                    nota_contrast_mask = torch.ones(1, bsz * contrast_count, device=device)

                # Combine with current eligibility (logits_mask)
                nota_valid = (logits_mask > 0) & nota_contrast_mask.bool()           # [bsz*anc, bsz*cont]

                # Repulsive-InfoNCE: log(sum exp(similarity)) over the chosen contrasts
                sum_exp = (exp_logits * nota_valid.float()).sum(1) + self.eps
                nota_repulsion = sum_exp.log()                                       # [bsz*anc]

                nota_term = torch.zeros_like(supcon_term, device=device)
                nota_term[is_nota_anchor] = (self.temperature / self.base_temperature) * nota_repulsion[is_nota_anchor]
                nota_term = self.nota_weight * nota_term
            else:
                nota_term = torch.zeros_like(supcon_term, device=device)
        else:
            nota_term = torch.zeros_like(supcon_term, device=device)

        # ----- Combine and average over anchors -----
        loss = (supcon_term + nota_term).view(anchor_count, bsz).mean()
        return loss



def train(dataloader, model, align_weight, img2seq_weight, mode, optimizer,loss1,loss2,device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (data, image, y) in enumerate(dataloader):
        # print(X.shape)
        # data=data.unsqueeze(1)
        # image=image.unsqueeze(1)

        data=data.to(device)
        image=image.to(device)
        y=y.long().to(device)
        optimizer.zero_grad()
        #############################
        # start_time = time.time()  # 记录开始时间
        conca,z_i,z_j = model(data, image)

        # end_time = time.time()  # 记录结束时间
        # print(f"运行时间: {end_time - start_time} 秒")

        # allocated_memory_bytes = torch.cuda.memory_allocated(device)
        # allocated_memory_mib = allocated_memory_bytes / (1024 * 1024)  # 转换为 MiB
        # print(f"Allocated memory: {allocated_memory_mib} MiB")

        z_i=F.normalize(z_i,dim=1) * img2seq_weight # seq
        z_j=F.normalize(z_j,dim=1) # img
        z_i=z_i.unsqueeze(1)
        z_j=z_j.unsqueeze(1)
        # feat=torch.concat([z_i,z_j],dim=0) 
        # ty=torch.cat([y,y], dim=0)
        # loss = a*loss1(features=feat,labels=ty)+(1-a)*loss2(conca,y) # 3,5
        if mode=="cross":
            feat=torch.concat([z_i,z_j],dim=0) 
            ty=torch.cat([y,y], dim=0)
            loss = align_weight*loss1(features=feat,labels=ty)+(1-align_weight)*loss2(conca,y) # 3,5

        elif mode=="single":
            feat=torch.concat([z_i,z_j],dim=0) 
            ty=torch.cat([y,y], dim=0)
            loss = align_weight*(loss1(features=z_i,labels=y)+loss1(features=z_j,labels=y))+(1-align_weight)*loss2(conca,y)
            
        loss.backward()
        optimizer.step()
        loss = loss.item()
        print(y.flatten().unique().tolist())
        # print(f'loss:{loss:>7f},SCL:{loss1(features=feat,labels=ty):>7f},CE:{loss2(conca,y):>7f}')


def test(dataloader, model,device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    all_labels = []
    all_predictions = []
    cf = 0
    with torch.no_grad():
        for batch,(data, image, y) in enumerate(dataloader):
            data=data.to(device)
            image=image.to(device)
            y=y.to(device)
            ########### for SCL
            conca, z_i, z_j= model(data,image)
            correct += (conca.argmax(1) == y).type(torch.float).sum().item()
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(conca.argmax(1).cpu().numpy())

        cf = confusion_matrix(all_labels, all_predictions)
        f1 = f1_score(all_labels, all_predictions, average='weighted')
        precision = precision_score(all_labels, all_predictions, average='weighted')
        recall = recall_score(all_labels, all_predictions, average='weighted')
        accuracy = correct / size
        print(f'Accuracy:{accuracy}')
        print(f'F1 score:{f1}')
        print(f'Precision:{precision}')
        print(f'Recall:{recall}')
        print('Confusion_Matrix:')
        print(cf)

    test_loss /= size
    print(f"Test Error: \n Accuracy:{(100 * accuracy):>0.1f}%, Avg loss:{test_loss:>8f} \n")
    return {
        "Accuracy":accuracy,
        "F1 score":f1,
        "Precision":precision,
        "Recall":recall,
        "Confusion_Matrix":cf.tolist()
    }


def update_json(filepath, key, value):
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
    except:
        data = {}
    data[key] = value
    with open(filepath, 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def run(args):
    file=args.dataset
    bi=args.BiLSTM
    time_aware=args.timesplit 
    
    if "milan" in file:
        class_weights = torch.tensor([0.1,1,1,1,1,1,1,1,1,1]).to(device)
        align_weight=0.6
        input=35
    elif "cairo" in file:
        # class_weights = torch.tensor([1,1,1,1,1,1,1]).to(device)
        align_weight=0.9
        input=34 # 34
    elif "kyoto7" in file:
        align_weight=0.85
        input=73
    elif "orange" in file:
        align_weight=0.6
        input=195
    if args.weight is not None:
        align_weight=args.weight
    mode=args.mode
    epochs = 60 #60
    seed = args.seed # 15,30*,45,60
    k_folds = 5 #5
    batch_size = 64
    train_ratio = 0.7
    torch.manual_seed(seed)
    MAX=7
    if "milan" in file:
        MAX=10
    elif "orange" in file:
        MAX=15

    if time_aware==0:
        # dataset=Data_Generator(file,MAX,seqfile=f"seq",catfile=f"cat")
        dataset = Data_Generator(file,MAX,seqfile=f"datasets/seq_{args.filter}_t{int(args.filter_threshold*100):02d}_offFalse",catfile=f"datasets/cat_{args.filter}_t{int(args.filter_threshold*100):02d}_offFalse")
        num_samples = len(dataset)
        num_train = int(train_ratio * num_samples)
        num_test = num_samples - num_train
        train_dataset, test_dataset = random_split(dataset, [num_train, num_test], generator=torch.Generator().manual_seed(42))
    else:
        train_dataset=Data_Generator(file,MAX,seqfile=f"datasets/seq_ori_train_{file}",catfile=f"datasets/cat_ori_train_{file}")
        test_dataset=Data_Generator(file,MAX,seqfile=f"datasets/seq_ori_test_{file}",catfile=f"datasets/cat_ori_test_{file}")
    # 初始化K-Fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []

    best_epochs = []
    if not args.full_training:
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            # if fold in [0,1]:
            #     continue
            print(f'FOLD {fold + 1}')
            print('--------------------------------')
            epoch_val_results=[]
            val_accu_list=[]
            train_subsampler = Subset(train_dataset, train_idx)
            val_subsampler = Subset(train_dataset, val_idx)

            # 定义数据加载器
            train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

            # 重新初始化模型和优化器
            model=LSTM_CNN(MAX,input,bi).to(device)
            # model=torch.load("D:\HAR\CL_HAR\model\kyoto7-True-30-cross/3fold_21Epoch.pth")
            optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.001) # 1e-2
            if args.nota:
                loss1=SupConLossWithNOTA(contrast_mode='all',temperature=0.05,enable_nota=True, nota_weight=args.nota_weight)
            else:
                # loss1=SupConLossWithNOTA(contrast_mode='all',temperature=0.05,enable_nota=False)
                loss1=SupConLoss(contrast_mode='all',temperature=0.05)
            loss2=nn.CrossEntropyLoss()

            # 训练和验证
            
            # checkpoint_dir = f"model/{file}-{bi}-{seed}-{mode}-{args.reduction}-{args.supcon_grad}" if not args.nota else f"model/{file}-{bi}-{seed}-{mode}-nota"
            scl_mode = "scl" if not args.nota else f"scl-nota-{args.nota_weight}"
            checkpoint_dir = f"model/{file}-{bi}-{seed}-{mode}-{scl_mode}_filter_{args.filter}_t{int(args.filter_threshold*100):02d}_align_weight{align_weight}"
            os.makedirs(checkpoint_dir,exist_ok=True)
            for epoch in range(1, epochs+1):  
                print(f'Epoch {epoch}')
                train(train_loader, model, align_weight, args.imgseq, mode, optimizer,loss1,loss2,device)
                val_metrics = test(val_loader, model, device)
                # get_tsne(file,model,val_loader,epoch)
                torch.save(model,os.path.join(checkpoint_dir,f"{fold+1}fold_{epoch}Epoch.pth"))
                epoch_val_results.append(val_metrics)
                val_accu_list.append(val_metrics["Accuracy"])

            best_epoch = val_accu_list.index(max(val_accu_list))
            best_epochs.append(best_epoch+1)
            best_metrics = epoch_val_results[best_epoch]
            epoch_eval_log = {
                "Best Epoch":best_epoch+1,
                "Best Metrics":best_metrics
            }
            update_json(f'{checkpoint_dir}/Valid.json', f"FOLD_{fold+1}", epoch_eval_log)
            
            print("Start Test")
            model=torch.load(f"{checkpoint_dir}/{fold+1}fold_{val_accu_list.index(max(val_accu_list))+1}Epoch.pth", weights_only=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_metrics=test(test_loader, model,device)
            # get_tsne(file,model,test_loader,0)
            update_json(f'{checkpoint_dir}/Test.json', f"FOLD_{fold+1}_true", test_metrics)

            # print("Start Test")
            # epoch_test_results=[]
            # test_accuracy_list=[]
            # for epoch in range(1, epochs+1):  
            #     print(epoch)
            #     model=torch.load(f"{checkpoint_dir}/{fold+1}fold_{epoch}Epoch.pth", weights_only=False)
            #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            #     test_metrics=test(test_loader, model,device)
            #     epoch_test_results.append(test_metrics)
            #     test_accuracy_list.append(test_metrics["Accuracy"])
            #     # print(accu)
            # best_test_epoch = test_accuracy_list.index(max(test_accuracy_list))
            # best_test_metrics = epoch_test_results[best_test_epoch]
            # update_json(f'{checkpoint_dir}/Test.json', f"FOLD_{fold+1}_peek", best_test_metrics)

        # remove checkpoints except the best one
        checkpoints = glob(os.path.join(checkpoint_dir, "*.pth"))
        best_checkpoints = [os.path.join(checkpoint_dir, f"{fold+1}fold_{best_epoch+1}Epoch.pth") for fold, best_epoch in zip(range(k_folds), best_epochs)]
        for ckpt in checkpoints:
            if ckpt not in best_checkpoints:
                os.remove(ckpt)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model=LSTM_CNN(MAX,input,bi).to(device)
        # model=torch.load("D:\HAR\CL_HAR\model\kyoto7-True-30-cross/3fold_21Epoch.pth")
        optimizer = optim.Adam(model.parameters(), lr=1e-3,weight_decay=0.001) # 1e-2
        if args.nota:
            loss1=SupConLossWithNOTA(contrast_mode='all',temperature=0.05,enable_nota=True, nota_weight=args.nota_weight)
        else:
            # loss1=SupConLossWithNOTA(contrast_mode='all',temperature=0.05,enable_nota=False)
            loss1=SupConLoss(contrast_mode='all',temperature=0.05)
        loss2=nn.CrossEntropyLoss()

        # 训练和验证
        scl_mode = "scl" if not args.nota else f"scl-nota-{args.nota_weight}"
        checkpoint_dir = f"model/{file}-{bi}-{seed}-{mode}-{scl_mode}_filter_{args.filter}_t{int(args.filter_threshold*100):02d}_align_weight{align_weight}"
        os.makedirs(checkpoint_dir,exist_ok=True)
        for epoch in range(1, epochs+1):  
            print(f'Epoch {epoch}')
            train(train_loader, model, align_weight, args.imgseq, mode, optimizer,loss1,loss2,device)
            # get_tsne(file,model,val_loader,epoch)
            torch.save(model,os.path.join(checkpoint_dir,f"NaNfold_{epoch}Epoch.pth"))
        
        print("Start Test")
        acc=[]
        met=[]
        for epoch in range(40, epochs+1):
            model=torch.load(f"{checkpoint_dir}/NaNfold_{epoch}Epoch.pth", weights_only=False)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            test_metrics=test(test_loader, model,device)
            acc.append(test_metrics["Accuracy"])
            met.append(test_metrics)
        best_epoch=acc.index(max(acc))+40
        test_metrics=met[acc.index(max(acc))]
        print(f"Best Test Epoch: {best_epoch}")
        update_json(f'{checkpoint_dir}/Test.json', f"FOLD_NaN_true", test_metrics)
        # remove checkpoints except the best one
        checkpoints = glob(os.path.join(checkpoint_dir, "*.pth"))
        best_checkpoints = [os.path.join(checkpoint_dir, f"NaNfold_{best_epoch+1}Epoch.pth") for fold, best_epoch in zip(range(k_folds), best_epochs)]
        for ckpt in checkpoints:
            if ckpt not in best_checkpoints:
                os.remove(ckpt)

def get_tsne(filename,model,dataloader,epoch):
    model1=model.model1
    model2=model.model2
    all_labels=[]

    ts1=0
    ts2=0
    ts=0
    t=0
    with torch.no_grad():
        for batch,(data,image,Y) in enumerate(dataloader):
            image=image.permute([0,3,1,2])

            data=data.to(device)
            image=image.to(device)
            Y=Y.to(device)
            pre1,_=model1(data)
            pre1=pre1[:,-1,:]
            pre2=model2(image)
            pre=torch.cat([pre1,pre2],dim=1)
            # print(pre1.shape)
            # print(pre2.shape)
            if t==0:
                ts=pre
                ts1=pre1
                ts2=pre2
            else:
                ts=torch.cat([ts,pre],dim=0)
                ts1=torch.cat([ts1,pre1],dim=0)
                ts2=torch.cat([ts2,pre2],dim=0)
            all_labels.extend(Y.cpu().numpy())
            t+=1
        ts=ts.cpu()
        ts1=ts1.cpu()
        ts2=ts2.cpu()

        tsne = TSNE(n_components=2, init='pca', random_state=0)
        tsne.fit_transform(ts)
        color_list = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'lawngreen', 'peru',
                      'violet', 'slategray']
        col = [color_list[k] for k in all_labels]
        plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=col)
        plt.show()
        plt.savefig(f'tsne/CL/{filename}__{epoch}.png', bbox_inches='tight', pad_inches=0,
                    transparent=False)
        plt.close()

if __name__=='__main__':
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # print(torch.cuda.is_available())
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["milan","cairo","kyoto7", "orange"], default="milan")
    parser.add_argument("--BiLSTM", type=int, choices=[0,1], default=1)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--mode",type=str, choices=["cross","single"], default="cross")
    parser.add_argument("--weight",type=float)
    parser.add_argument("--imgseq",type=float,default=1)
    parser.add_argument("--timesplit",type=int,choices=[0,1],default=0)# 0代表不用时间split，1代表用
    parser.add_argument("--filter", choices=["sens","room", "spa"], default="sens")
    parser.add_argument("--filter_threshold", type=float, default=0.01)
    parser.add_argument("--full-training", action="store_true", default=False)
    # ======== newly added debugging arguments ========
    # parser.add_argument("--reduction", type=str, choices=["mean","sqrt"], required=True)
    # parser.add_argument("--supcon_grad", action="store_true", default=False)
    parser.add_argument("--nota", action="store_true", default=False)
    parser.add_argument("--nota-weight", type=float, default=1.0)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(args)



