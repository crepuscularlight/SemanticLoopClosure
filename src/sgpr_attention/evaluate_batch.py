from configs.config_loader import model_config
from src.dataset import get_dataset
from src.trainer import get_trainer
from src.models import get_model

import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import time
from glob import glob


def sys_parse():
    parser = argparse.ArgumentParser(description='system parse')
    parser.add_argument("--config",default="./configs/sgpr_baseline.yml",required=False)
    parser.add_argument("--version",default=None,required=False)
    args = parser.parse_args()

    return args

def main():
    args = sys_parse()
    cfg=model_config()
    cfg.load(args.config)

    tmp=os.path.join("./experiments",cfg.exp_name,args.version)
    ckpt_path=tmp+"/best.pth"

    
    # print(ckpt_path)

    model = get_model()[cfg.model](cfg)
    state_dict = torch.load(ckpt_path)
    model.load_state_dict(state_dict)

    model.eval()
    model.zero_grad()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.zero_grad()

    output_dir = os.path.join(
        "experiments_out", cfg.exp_name, args.version,"test_result"
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    dataset = get_dataset()[cfg.dataset]

    test_sequences=cfg.test_sequences
    for sq in test_sequences:

        cfg.test_sequences=[str(sq)]
        test_dataset=dataset(cfg,mode="test")
        test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=64,shuffle=False,collate_fn=test_dataset.remove_ambiguity)



        gt_db = []
        pred_db = []

        for batch in tqdm(test_loader):

            if batch is None:
                continue
            pred_batch,_,_=model(batch)


            gt_db.extend(batch["target"].cpu().detach().numpy())

            pred_db.extend(pred_batch.cpu().detach().numpy())
            print(pred_batch)
            print(batch["target"])


        pred_db = np.array(pred_db)
        gt_db = np.array(gt_db).astype(int)


        sequence=cfg.test_sequences[0]
        gt_db_path = os.path.join(output_dir, sequence + "_gt_db.npy")
        pred_db_path = os.path.join(output_dir, sequence + "_DL_db.npy")

        np.save(gt_db_path, gt_db)
        np.save(pred_db_path, pred_db)

        fpr, tpr, roc_thresholds = metrics.roc_curve(gt_db, pred_db)

        roc_auc = metrics.auc(fpr, tpr)
        # print("fpr: ", fpr)
        # print("tpr: ", tpr)
        # print("thresholds: ", roc_thresholds)
        # print("roc_auc: ", roc_auc)

        # plot ROC Curve
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('DL ROC Curve')
        plt.legend(loc="lower right")
        roc_out = os.path.join(output_dir, sequence + "_DL_roc_curve.png")
        plt.savefig(roc_out)

        #### P-R
        precision, recall, pr_thresholds = metrics.precision_recall_curve(gt_db, pred_db)
        # plot p-r curve
        plt.figure()
        lw = 2
        plt.plot(recall, precision, color='darkorange',
                 lw=lw, label='P-R curve')
        plt.axis([0, 1, 0, 1.2])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('DL Precision-Recall Curve')
        plt.legend(loc="lower right")
        pr_out = os.path.join(output_dir, sequence + "_DL_pr_curve.png")
        plt.savefig(pr_out)

        # plt.show()
        # calc F1-score
        F1_score = 2 * precision * recall / (precision + recall)
        F1_score = np.nan_to_num(F1_score)
        F1_max_score = np.max(F1_score)
        f1_out = os.path.join(output_dir, sequence + "_DL_F1_max.txt")
        print('F1 max score', F1_max_score)
        with open(f1_out, "w") as out:
            out.write(str(F1_max_score))

if __name__=="__main__":
    main()
