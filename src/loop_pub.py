#!/usr/bin/env python3

# from re import L
import rospy
import rospkg
import numpy as np
import os
import shutil
from glob import glob
from iscloam.msg import LoopInfo


def loop_callback(msg,args):
    output_name=args
    # print(output_name)
    if(len(msg.matched_id)==0):
        return 
    if os.path.exists(output_name):
        current_array=np.loadtxt(output_name)
    else:
        current_array=np.empty((0,2))
    print(msg.matched_id)
    new_row=np.array([[msg.current_id,msg.matched_id[0]]])
    update_array=np.vstack((current_array,new_row))
    np.savetxt(output_name,update_array)

if __name__=="__main__":
    rospy.init_node("pub_loop")
    r = rospkg.RosPack()
    path=r.get_path("iscloam")

    output=os.path.join(path,"test")
    if not os.path.exists(output):
        os.mkdir(output)
    output=output+"/c_loop.txt"
    if os.path.exists(output):
        os.remove(output)
    sub=rospy.Subscriber("/loop_closure",LoopInfo,loop_callback,(output))
    rospy.spin()