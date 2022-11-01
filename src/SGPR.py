#!/home/liudiyang/Application/miniconda3/envs/sgpr/bin/python

from sgpr_attention.configs.config_loader import model_config
from sgpr_attention.src.dataset import get_dataset
from sgpr_attention.src.trainer import get_trainer
from sgpr_attention.src.models import get_model

import torch
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from tqdm import tqdm
import time
from glob import glob
from threading import Thread, Lock
from math import sqrt

import rospy
import rospkg
import std_msgs
from nav_msgs.msg import Odometry

class SGPR:
    def __init__(self):
        r = rospkg.RosPack()
        path=r.get_path("iscloam")
        sgpr_path=os.path.join(path,"src/sgpr_attention/")

        cfg=model_config()
        cfg.load(os.path.join(sgpr_path,"configs/sgpr_geo_attention.yml"))
        # cfg.test_pairs_dir=args.test


        tmp=os.path.join(sgpr_path,"experiments",cfg.exp_name,"all_dp0.05")
        ckpt_path=tmp+"/best.pth"

        self.model = get_model()[cfg.model](cfg)
        state_dict = torch.load(ckpt_path)
        self.model.load_state_dict(state_dict)


        # self.model.eval()
        # self.model.zero_grad()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device)
        # self.model.eval()
        # self.model.zero_grad()

        # output_dir = os.path.join(
        #     "ros", cfg.exp_name, "all_test",
        # )

        # if not os.path.exists(output_dir):
        #     os.makedirs(output_dir)

        self.dataset = get_dataset()[cfg.dataset]
        self.test_dataset=self.dataset(cfg,mode="test")



        self.travel_distance_arr=[]
        self.pos_arr=[]
        self.matched_frame_id=[]
    
    def loop_detection(self,odom_in):
        # current_t=odom_in.translation()
        current_t=odom_in
        if(len(self.travel_distance_arr)==0):
            self.travel_distance_arr.append(0)
        else:
            # double dis_temp = travel_distance_arr.back()+std::sqrt((pos_arr.back()-current_t).array().square().sum())
            dis_temp=self.travel_distance_arr[-1]+np.sqrt(np.sum(np.square(self.pos_arr[-1]-current_t),axis=0))
            self.travel_distance_arr.append(dis_temp)
        self.pos_arr.append(current_t)
        
        self.current_frame_id=len(self.pos_arr)-1
        self.matched_frame_id=[]
        best_matched_id=0
        
        check_frame_list=[]
        #select frames
        for i in range(len(self.pos_arr)):
            delta_travel_distance=self.travel_distance_arr[-1]-self.travel_distance_arr[i]
            pos_distance=np.sqrt(np.sum(np.square(self.pos_arr[i]-self.pos_arr[-1]),axis=0))
            if delta_travel_distance>20 and pos_distance<delta_travel_distance*0.02 :
                check_frame_list.append(i)
        if len(check_frame_list)>64:
            check_frame_list=np.random.choice(check_frame_list,64,replace=False)
            check_frame_list=list(check_frame_list)
        
        self.test_dataset.set_frames(check_frame_list,self.current_frame_id)
        test_loader=torch.utils.data.DataLoader(self.test_dataset,batch_size=64,shuffle=False,collate_fn=self.test_dataset.remove_ambiguity)
        self.model.eval()
        self.model.zero_grad()
        pred_db=[]
        for batch in (test_loader):

            if batch is None:
                continue

            pred_batch,_,_=self.model(batch)

            pred_db.extend(pred_batch.cpu().detach().numpy())

        pred_db = np.array(pred_db)
        if pred_db.shape[0]==0:
            return 
        best_id=np.argmax(pred_db)
        idx = np.argsort(pred_db, axis=0)[-3:].astype(int)
        if np.min(pred_db[idx])<0.999:
            return

        best_matched_id=[check_frame_list[int(i)] for i in idx]

        

        for item in best_matched_id:
            if(best_matched_id!=0):
                # print("current:",self.current_frame_id,"match:",item)
                self.matched_frame_id.append(item)


        

