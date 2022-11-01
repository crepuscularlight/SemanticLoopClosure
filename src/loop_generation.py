#!/usr/bin/env python3

import rospy
import rospkg
import numpy as np
import os
from glob import glob
from iscloam.msg import LoopInfo


if __name__=="__main__":
    rospy.init_node("gnn_loop")
    loop_pub=rospy.Publisher("loop_closure",LoopInfo,queue_size=10)

    graphs_dir="/home/liudiyang/ms/refer/sgpr_imitate/data_preprocess/debug_dgcnn_bbox/graphs"
    sequence="00"
    graphs_folder=os.path.join(graphs_dir,sequence+"/*")
    files=sorted(glob(graphs_folder))

    # path=os.path.abspath(".")
    r = rospkg.RosPack()
    path=r.get_path("iscloam")

    check_window=20

    pair_path=os.path.join(path,"pairs")
    if(not os.path.exists(pair_path)):
        os.mkdir(pair_path)
    pair_file=pair_path+f"/{sequence}.txt"
    loop_msgs=np.array((0,2))
    for i in range(0,100):
        msg=LoopInfo()
        if(i<=check_window):
            loop_msgs=np.vstack((loop_msgs,np.array([i,i]).reshape((-1,2))))
        else:

            with open(pair_file,"w") as f:
                for j in range(0,i-check_window):
                    f.writelines(sequence+"/"+str(i).zfill(6)+".npz "+sequence+"/"+str(j).zfill(6)+".npz"+"\n")
            cmd=f"/home/liudiyang/ms/refer/sgpr_new0/evaluate_ros.py --config /home/liudiyang/ms/refer/sgpr_new0/configs/sgpr_single.yml --test {pair_path} --version base"
            os.system(cmd)
            pred=np.load(os.path.join(pair_path,sequence+"_DL_db.npy"))
            # msg.header.stamp=rospy.Time.now()
            msg.current_id=i
            msg.matched_id=np.argmax(pred)
            loop_msgs=np.vstack((loop_msgs,np.array([msg.current_id,msg.matched_id]).reshape((-1,2))))
        # loop_pub.publish(msg)
        # rospy.loginfo("pub one")
    
    loop_msgs=loop_msgs.astype(int)
    np.savetxt(pair_path+f"/loop_{sequence}.txt",loop_msgs)
