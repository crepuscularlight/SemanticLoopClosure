#!/usr/bin/env python3

# from re import L
import rospy
import rospkg
import numpy as np
import os
import shutil
from glob import glob
from iscloam.msg import LoopInfo

if __name__=="__main__":
    rospy.init_node("gtRelativePose")
    r = rospkg.RosPack()
    path=r.get_path("iscloam")
    trajectory=os.path.join(path,"trajectory/00.txt")
    poses=np.loadtxt(trajectory)