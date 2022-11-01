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
import queue

import rospy
import rospkg
import std_msgs
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from SGPR import SGPR
from iscloam.msg import LoopInfo

detector=SGPR()
model=detector.model

loop_info_pub=rospy.Publisher("loop_closure_python",LoopInfo,queue_size=100)



odom_buf=queue.Queue()
pointclouds_buf=queue.Queue()
mutex=Lock()
def odom_callback(msg):
    mutex.acquire()
    odom_buf.put(msg)
    mutex.release()
def velodyne_callback(msg):
    mutex.acquire()
    pointclouds_buf.put(msg)
    mutex.release()

def loop_detection():
    while True:
        if(odom_buf.qsize()>0 and pointclouds_buf.qsize()>0):
            mutex.acquire()
            odom_msg=odom_buf.get()
            pointcloud_msg=pointclouds_buf.get()
            pointcloud_time=pointcloud_msg.header.stamp
            odom_in=np.ones((3,1))
            odom_in[0]=odom_msg.pose.pose.position.x
            odom_in[1]=odom_msg.pose.pose.position.y
            odom_in[2]=odom_msg.pose.pose.position.z
            mutex.release()
            detector.loop_detection(odom_in)

            loop=LoopInfo()
            loop.header.stamp=pointcloud_time
            loop.header.frame_id="velodyne"
            loop.current_id=detector.current_frame_id
            for i in range(0,len(detector.matched_frame_id)):
                loop.matched_id.append(detector.matched_frame_id[i])
            loop_info_pub.publish(loop)



if __name__=="__main__":
    rospy.init_node("loop_generation")
    odom_sub=rospy.Subscriber("odom",Odometry,odom_callback)
    velodyne_sub=rospy.Subscriber("/velodyne_points_filtered",PointCloud2,velodyne_callback)
    Thread(target=loop_detection, daemon=True).start()

    rospy.spin()

    


