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
# from sklearn.manifold import TSNE
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

    def forward(self,x):
        out,_ = self.lstm(x)
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
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(512, 32),
            )
            self.mlp3 = nn.Sequential(
                nn.Linear(512+20, MAX)
            )
        elif bi==1:
            self.mlp1 = nn.Sequential(
                nn.Linear(20*2,32), 
            )
            self.mlp2 = nn.Sequential(
                nn.Linear(512, 32),
                # nn.Linear(20, 32)
            )
            self.mlp3 = nn.Sequential(
                nn.Linear(512+20*2, MAX)
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
        
        conca = torch.cat([seq, image], dim=1)
        conca = self.mlp3(conca)

        seq = self.mlp1(seq)
        image = self.mlp2(image)
        return conca,seq,image


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
        loss = loss.view(anchor_count, batch_size).mean()
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

        # start_time = time.time() 
        conca,z_i,z_j = model(data, image)

        # end_time = time.time() 
        # print(f"Running Time Per Batch: {end_time - start_time} 秒")

        # allocated_memory_bytes = torch.cuda.memory_allocated(device)
        # allocated_memory_mib = allocated_memory_bytes / (1024 * 1024) 
        # print(f"Allocated memory: {allocated_memory_mib} MiB")

        z_i=F.normalize(z_i,dim=1) * img2seq_weight # seq
        z_j=F.normalize(z_j,dim=1) # img
        z_i=z_i.unsqueeze(1)
        z_j=z_j.unsqueeze(1)

        if mode=="cross":
            feat=torch.concat([z_i,z_j],dim=0) 
            ty=torch.cat([y,y], dim=0)
            loss = align_weight*loss1(features=feat,labels=ty)+(1-align_weight)*loss2(conca,y) 

        elif mode=="single":
            feat=torch.concat([z_i,z_j],dim=0) 
            ty=torch.cat([y,y], dim=0)
            loss = align_weight*(loss1(features=z_i,labels=y)+loss1(features=z_j,labels=y))+(1-align_weight)*loss2(conca,y)
            
        loss.backward()
        optimizer.step()
        loss = loss.item()
        print(y.flatten().unique().tolist())


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
        align_weight=0.6
        input=35
    elif "cairo" in file:
        align_weight=0.9
        input=34 # 34
    elif "kyoto7" in file:
        align_weight=0.85
        input=73
    if args.weight is not None:
        align_weight=args.weight
        
    mode=args.mode
    epochs = args.epoch 
    seed = args.seed 
    k_folds = args.K 
    batch_size = args.batchsize
    train_ratio = args.trainratio
    lr=args.lr
    torch.manual_seed(seed)
    MAX=7

    if not args.ckpt is None:
        print("Skip Validation")
        args.full_training = True
        
    if "milan" in file:
        MAX=10

    if time_aware==0:
        dataset = Data_Generator(file,MAX,seqfile=f"datasets/seq_{args.filter}_bin{int(args.bin)}_t{int(args.filter_threshold*100):02d}",catfile=f"datasets/cat_{args.filter}_t{int(args.filter_threshold*100):02d}")
        num_samples = len(dataset)
        num_train = int(train_ratio * num_samples)
        num_test = num_samples - num_train
        train_dataset, test_dataset = random_split(dataset, [num_train, num_test], generator=torch.Generator().manual_seed(42))
    else:
        train_dataset=Data_Generator(file,MAX,seqfile=f"datasets/seq_ori_train_{file}",catfile=f"datasets/cat_ori_train_{file}")
        test_dataset=Data_Generator(file,MAX,seqfile=f"datasets/seq_ori_test_{file}",catfile=f"datasets/cat_ori_test_{file}")
        
    # K-Fold
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    results = []

    # Loss
    loss1=SupConLoss(contrast_mode='all',temperature=0.05)
    loss2=nn.CrossEntropyLoss()
    
    best_epochs = []
    if not args.full_training:
        for fold, (train_idx, val_idx) in enumerate(kf.split(train_dataset)):
            print(f'FOLD {fold + 1}')
            print('--------------------------------')
            epoch_val_results=[]
            val_accu_list=[]
            train_subsampler = Subset(train_dataset, train_idx)
            val_subsampler = Subset(train_dataset, val_idx)

            train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=batch_size, shuffle=False)

            model=LSTM_CNN(MAX,input,bi).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001) 
            
            scl_mode = "scl" if not args.nota else f"scl-nota-{args.nota_weight}"
            checkpoint_dir = f"model/{file}-{bi}-{seed}-{mode}-{scl_mode}_filter_{args.filter}_t{int(args.filter_threshold*100):02d}_align_weight{align_weight}"
            os.makedirs(checkpoint_dir,exist_ok=True)
            for epoch in range(1, epochs+1):  
                print(f'Epoch {epoch}')
                train(train_loader, model, align_weight, args.imgseq, mode, optimizer,loss1,loss2,device)
                val_metrics = test(val_loader, model, device)
                # get_(file,model,val_loader,epoch)
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
            # get_(file,model,test_loader,0)
            update_json(f'{checkpoint_dir}/Test.json', f"FOLD_{fold+1}_true", test_metrics)
            
        # remove checkpoints except the best one
        checkpoints = glob(os.path.join(checkpoint_dir, "*.pth"))
        best_checkpoints = [os.path.join(checkpoint_dir, f"{fold+1}fold_{best_epoch+1}Epoch.pth") for fold, best_epoch in zip(range(k_folds), best_epochs)]
        for ckpt in checkpoints:
            if ckpt not in best_checkpoints:
                os.remove(ckpt)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        model=LSTM_CNN(MAX,input,bi).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.001)

        # Train and Validation
        scl_mode = "scl" if not args.nota else f"scl-nota-{args.nota_weight}"
        checkpoint_dir = f"model/{file}-{bi}-{seed}-{mode}-{scl_mode}_filter_{args.filter}_t{int(args.filter_threshold*100):02d}_align_weight{align_weight}"
        os.makedirs(checkpoint_dir,exist_ok=True)
        if args.ckpt is None:
            for epoch in range(1, epochs+1):  
                print(f'Epoch {epoch}')
                train(train_loader, model, align_weight, args.imgseq, mode, optimizer,loss1,loss2,device)
                # get_(file,model,val_loader,epoch)
                torch.save(model,os.path.join(checkpoint_dir,f"NaNfold_{epoch}Epoch.pth"))
        
        print("Start Test")
        acc=[]
        met=[]
        if args.ckpt is None:
            model=torch.load(f"{checkpoint_dir}/NaNfold_{epochs}Epoch.pth", weights_only=False)
        else:
            model=torch.load(args.ckpt, weights_only=False)
            
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        test_metrics=test(test_loader, model,device)
        acc.append(test_metrics["Accuracy"])
        met.append(test_metrics)
        best_epoch=acc.index(max(acc))+40
        test_metrics=met[acc.index(max(acc))]
        print(f"Best Test Epoch: {best_epoch}")
        update_json(f'{checkpoint_dir}/Test.json', f"FOLD_NaN_true", test_metrics)
        
        # remove checkpoints except the best one
        if args.ckpt is None:
            checkpoints = glob(os.path.join(checkpoint_dir, "*.pth"))
            best_checkpoints = [os.path.join(checkpoint_dir, f"NaNfold_{best_epoch+1}Epoch.pth") for fold, best_epoch in zip(range(1), best_epochs)]
            for ckpt in checkpoints:
                if ckpt not in best_checkpoints:
                    os.remove(ckpt)

# def get_tsne(filename,model,dataloader,epoch):
#     model1=model.model1
#     model2=model.model2
#     all_labels=[]

#     ts1=0
#     ts2=0
#     ts=0
#     t=0
#     with torch.no_grad():
#         for batch,(data,image,Y) in enumerate(dataloader):
#             image=image.permute([0,3,1,2])

#             data=data.to(device)
#             image=image.to(device)
#             Y=Y.to(device)
#             pre1,_=model1(data)
#             pre1=pre1[:,-1,:]
#             pre2=model2(image)
#             pre=torch.cat([pre1,pre2],dim=1)
#             # print(pre1.shape)
#             # print(pre2.shape)
#             if t==0:
#                 ts=pre
#                 ts1=pre1
#                 ts2=pre2
#             else:
#                 ts=torch.cat([ts,pre],dim=0)
#                 ts1=torch.cat([ts1,pre1],dim=0)
#                 ts2=torch.cat([ts2,pre2],dim=0)
#             all_labels.extend(Y.cpu().numpy())
#             t+=1
#         ts=ts.cpu()
#         ts1=ts1.cpu()
#         ts2=ts2.cpu()

#         tsne = TSNE(n_components=2, init='pca', random_state=0)
#         tsne.fit_transform(ts)
#         color_list = ['black', 'red', 'green', 'blue', 'yellow', 'purple', 'lawngreen', 'peru',
#                       'violet', 'slategray']
#         col = [color_list[k] for k in all_labels]
#         plt.scatter(tsne.embedding_[:, 0], tsne.embedding_[:, 1], c=col)
#         plt.show()
#         plt.savefig(f'tsne/CL/{filename}__{epoch}.png', bbox_inches='tight', pad_inches=0,
#                     transparent=False)
#         plt.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=["milan","cairo","kyoto7", "orange"], default="milan")
    parser.add_argument("--BiLSTM", type=int, choices=[0,1], default=1)
    parser.add_argument("--seed", type=int, default=30)
    parser.add_argument("--mode",type=str, choices=["cross","single"], default="cross")
    parser.add_argument("--weight",type=float, default=None)
    parser.add_argument("--K",type=int,help="K-fold cross validation")
    parser.add_argument("--imgseq",type=float,default=1)
    parser.add_argument("--timesplit",type=int,choices=[0,1],default=0)
    parser.add_argument("--filter", choices=["sens","room", "spa"], default="sens")
    parser.add_argument("--filter_threshold", type=float, default=0.01)
    parser.add_argument("--bin", type=int, default=1)
    parser.add_argument("--full-training", action="store_true", default=False)
    
    parser.add_argument("--epoch", ,type=int, default=60)
    parser.add_argument("--batchsize", ,type=int, default=64)
    parser.add_argument("--trainratio", type=float, default=0.7, help="Fraction of data used for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--ckpt", type=str, help="Pretrained Model", default= None)
    
    
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run(args)









