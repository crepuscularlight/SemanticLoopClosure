import torch
import argparse
from tqdm import tqdm,trange
import os
import wandb
from datetime import datetime

from configs.config_loader import model_config
from src.dataset import get_dataset
from src.trainer import get_trainer
from src.models import get_model

def sys_parse():
    parser = argparse.ArgumentParser(description='system parse')
    parser.add_argument("--config",default="./configs/sgpr_baseline.yml",required=False)
    parser.add_argument('--resume', action='store_true',help='Resume from ckpt.')
    parser.add_argument("--project",default="sgpr-test-debug",required=False)
    parser.add_argument("--version",default=None,required=False)
    parser.add_argument('--debug', action='store_true', help='debug.')
    args = parser.parse_args()

    return args

def main():
    args = sys_parse()
    cfg=model_config()
    cfg.load(args.config)

    time_stamp = datetime.now().strftime('%m-%d_%H-%M-%S')
    if args.version is None:
        args.version = time_stamp

    if args.debug:
        cfg.exp_name="sgpr-debug"

    output_dir = os.path.join(
        "experiments", cfg.exp_name, args.version,
    )


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not args.debug:
        wandb.init(project=cfg.exp_name,name=args.version ,entity="liudyang")

    model=get_model()[cfg.model](cfg)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    scheduler=torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma=0.97)
    epoch_start = 0

    if args.resume:
        checkpoint = torch.load(os.path.join(output_dir, "latest.pth"))
        model.load_state_dict(checkpoint['model_state_dict'], strict=True)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    dataset=get_dataset()[cfg.dataset]

    train_dataset=dataset(cfg)
    val_dataset=dataset(cfg,"eval")

    train_loader=torch.utils.data.DataLoader(train_dataset, batch_size=cfg.batch_size,shuffle=True)
    val_loader=torch.utils.data.DataLoader(val_dataset, batch_size=cfg.batch_size,shuffle=True)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trainer=get_trainer()[cfg.trainer](cfg,model,optimizer,scheduler,device)


    epochs = trange(epoch_start, cfg.epoch, leave=True, desc="Epoch")
    iter=0
    f1_history_max=0
    for epoch in epochs:
        for batch in tqdm(train_loader):
            iter+=len(batch)
            loss=trainer.train_step(batch)

            if not args.debug:
                wandb.log({"train loss": loss})
            epochs.set_description("Epoch (Loss=%g)" % round(loss, 5))

        loss_eval,f1_score,pr_auc,accuracy_max=trainer.evaluate(val_loader)
        if not args.debug:
            wandb.log({"eval loss":loss_eval})
            wandb.log({"f1 score":f1_score})
            wandb.log({"pr_auc":pr_auc})
            wandb.log({"accuracy max":accuracy_max})


        dict_name = output_dir + "/" + str(epoch) + '.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            "scheduler_state_dict": scheduler.state_dict()
        }, dict_name)

        latest_dict_name =output_dir + "/" + 'latest.pth'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            "scheduler_state_dict": scheduler.state_dict()
        }, latest_dict_name)

        if f1_history_max <= f1_score:
            f1_history_max = f1_score
            dict_name = output_dir + "/" + str(epoch) + "_best" + '.pth'
            best_name=output_dir+"/"+"best.pth"
            torch.save(model.state_dict(), dict_name)
            torch.save(model.state_dict(),best_name)
            print("\n best model saved ", dict_name)
        print("------------------------------")
        # scheduler.step()
if __name__=="__main__":
    main()