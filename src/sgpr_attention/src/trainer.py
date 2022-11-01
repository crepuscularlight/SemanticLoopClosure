from tqdm import tqdm
from sklearn import metrics
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def get_trainer():
    return {"SGPRTrainer":SGPRTrainer}


class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=1.5, alpha=0.25, reduction='sum'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.eps=1e-6

    def forward(self, pt, target):
        pt=torch.clamp(pt, min=self.eps, max=1 - self.eps)
        loss = - (1 - pt) ** self.gamma * target * torch.log(pt) - \
            pt ** self.gamma * (1 - target) * torch.log(1 - pt)
        if self.alpha:
            alpha_t = self.alpha * target + (1 - self.alpha) * (1 - target)
            loss = alpha_t * loss
        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss


class SGPRTrainer:
    def __init__(self,cfg,model,optimizer=None,scheduler=None,device=None):
        self.model=model.to(device)
        self.optimizer=optimizer
        self.scheduler=scheduler
        self.device=device

        self.cfg=cfg
        self.criterion=nn.BCELoss()
        # self.criterion=FocalLoss()
    def train_step(self, data):

        self.model.train()
        self.optimizer.zero_grad()
        loss = self.compute_loss(data)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def compute_loss(self,data):

        prediction, _, _ = self.model(data)
        gt=data["target"].type(torch.FloatTensor).cuda(self.device)
        print("prediction shape", prediction)
        print("ground truth", gt)
        # prediction=torch.logit(prediction)
        # losses = torch.nn.functional.binary_cross_entropy(prediction, gt)
        losses=self.criterion(prediction,gt)
        return losses

    def evaluate(self,val_loader):
        pred_db = []
        gt_db = []
        losses=0
        for batch in tqdm(val_loader):
            loss_score, pred_b, gt_b = self.eval_step(batch)
            losses += loss_score
            pred_db.extend(pred_b)
            gt_db.extend(gt_b)

        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)

        # F1_score=metrics.f1_score(np.array(gt_db,dtype=int),pred_db)
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        F1_max_score = np.max(F1_score)
        print("test",F1_score)
        print("\nModel " +  " F1_max_score: " + str(F1_max_score) + ".")
        model_loss = losses / len(val_loader)
        print("\nModel " + " loss: " + str(model_loss) + ".")

        
        pr_auc=metrics.auc(recall,precision)

        thresholds=np.arange(0.1,1,0.1)
        accuracy_scores=[]
        for thresh in thresholds:
            accuracy_scores.append(metrics.accuracy_score(gt_db, (pred_db>thresh).astype(int)))
        
        accuracy_max= np.max(np.array(accuracy_scores))

        print("accuracy max",accuracy_max)

        return model_loss , F1_max_score, pr_auc , accuracy_max

    def eval_step(self, data):

        self.model.eval()
        prediction,_,_=self.model(data)
        gt = data["target"].type(torch.FloatTensor)#to(self.device)

        # losses = torch.nn.functional.binary_cross_entropy(prediction,gt.cuda(self.device))
        losses=self.criterion(prediction,gt.cuda(self.device))
        pred_batch = prediction.cpu().detach().numpy().reshape(-1)
        gt_batch = data["target"].cpu().detach().numpy().reshape(-1)

        return losses.item(),pred_batch,gt_batch

if __name__=="__main__":
    x=torch.rand([1,10]).cuda()
    y=torch.ones([1,10]).cuda()
    loss=FocalLossV1(gamma=2)
    out=loss(x,y)
    print(out)
