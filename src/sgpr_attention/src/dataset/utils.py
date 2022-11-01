import os
import numpy as np
import math


calib1=np.array( [4.276802385584e-04, -9.999672484946e-01, -8.084491683471e-03, -1.198459927713e-02, -7.210626507497e-03, 8.081198471645e-03, -9.999413164504e-01, -5.403984729748e-02, 9.999738645903e-01, 4.859485810390e-04, -7.206933692422e-03, -2.921968648686e-0])
calib2=np.array([2.347736981471e-04, -9.999441545438e-01, -1.056347781105e-02, -2.796816941295e-03, 1.044940741659e-02, 1.056535364138e-02, -9.998895741176e-01, -7.510879138296e-02, 9.999453885620e-01, 1.243653783865e-04, 1.045130299567e-02, -2.721327964059e-0])
calib3=np.array([-1.857739385241e-03, -9.999659513510e-01, -8.039975204516e-03, -4.784029760483e-03, -6.481465826011e-03, 8.051860151134e-03, -9.999466081774e-01, -7.337429464231e-02, 9.999773098287e-01, -1.805528627661e-03, -6.496203536139e-03, -3.339968064433e-01])
velo2camera_matrix=np.vstack((calib1,calib2,calib3))

def load_ros_pair(graph_pairs_dir,test_squence,check_frame_id,current_frame_id):
  paires=[]
  for item in check_frame_id:
    paires.append([os.path.join(graph_pairs_dir,test_squence,str(item).zfill(6)+".npz"),os.path.join(graph_pairs_dir,test_squence,str(current_frame_id).zfill(6)+".npz")])
  return paires
def load_paires(file,graph_pairs_dir):
  paires=[]
  with open(file) as f:
    while True:
      line=f.readline()
      if not line:
        break
      line=line.strip().split(" ")
      paires.append([os.path.join(graph_pairs_dir,line[0]),os.path.join(graph_pairs_dir,line[1])])
  return paires

def process_pair(path):
  data1 = np.load(path[0], allow_pickle=True)
  data2 = np.load(path[1], allow_pickle=True)

  data = {}

  pose1 = data1["pose"]
  pose2 = data2["pose"]

  data["nodes_1"] = data1["nodes"]
  data["nodes_2"] = data2["nodes"]


  dis = math.sqrt((pose1[3] - pose2[3]) ** 2 + (pose1[11] - pose2[11]) ** 2)

  data["pcn_features_1"] = data1["pcn_feature"]
  data["pcn_features_2"] = data2["pcn_feature"]



  data["centers_1"] = data1["centers"]
  data["centers_2"] = data2["centers"]

  data["bbox_1"]=data1["bbox"]
  data["bbox_2"]=data1["bbox"]

  data["distance"] = dis


  return data

def process_pair_points(path):
  """
  points [N,M,3]
  :param path:
  :return:
  """
  data1 = np.load(path[0], allow_pickle=True)
  data2 = np.load(path[1], allow_pickle=True)

  data = {}

  pose1 = data1["pose"]
  pose2 = data2["pose"]

  data["nodes_1"] = data1["nodes"]
  data["nodes_2"] = data2["nodes"]


  dis = math.sqrt((pose1[3] - pose2[3]) ** 2 + (pose1[11] - pose2[11]) ** 2)

  # data["pcn_features_1"] = data1["pcn_feature"]
  # data["pcn_features_2"] = data2["pcn_feature"]
  data["points_1"]=data1["normalized_points"]
  data["points_2"]=data2["normalized_points"]

  data["centers_1"] = data1["centers"]
  data["centers_2"] = data2["centers"]

  data["distance"] = dis


  return data

def process_pair_global_center(path):
  data1 = np.load(path[0], allow_pickle=True)
  data2 = np.load(path[1], allow_pickle=True)
  sequence=int(path[0].split(os.sep)[-2])

  if sequence <3:
    velo2camera=velo2camera_matrix[0]
  elif sequence==3:
    velo2camera=velo2camera_matrix[1]
  else:
    velo2camera=velo2camera_matrix[2]
  velo2camera=velo2camera.reshape(3,4)
  data = {}

  pose1 = data1["pose"]
  pose2 = data2["pose"]

  data["nodes_1"] = data1["nodes"]
  data["nodes_2"] = data2["nodes"]


  dis = math.sqrt((pose1[3] - pose2[3]) ** 2 + (pose1[11] - pose2[11]) ** 2)

  data["centers_1"] = data1["centers"]
  data["centers_2"] = data2["centers"]

  transform_matrix1=np.array(pose1).reshape(3,4)
  transform_matrix2=np.array(pose2).reshape(3,4)

  data["pcn_features_1"] = (transform_matrix1[:3,:3]@velo2camera[:3,:3]@data1["centers"].T+transform_matrix1[:3,:3]@np.expand_dims(velo2camera[:,3],axis=1)+np.expand_dims(transform_matrix1[:,3],axis=1)).T
  data["pcn_features_2"] = (transform_matrix2[:3,:3]@velo2camera[:3,:3]@data2["centers"].T+transform_matrix2[:3,:3]@np.expand_dims(velo2camera[:,3],axis=1)+np.expand_dims(transform_matrix2[:,3],axis=1)).T

  data["pcn_features_1"]=np.expand_dims(data["pcn_features_1"],axis=1)
  data["pcn_features_2"]=np.expand_dims(data["pcn_features_2"],axis=1)

  data["distance"] = dis

  return data


def process_pair_bbox(path):
  data1 = np.load(path[0], allow_pickle=True)
  data2 = np.load(path[1], allow_pickle=True)

  data = {}

  pose1 = data1["pose"]
  pose2 = data2["pose"]

  data["nodes_1"] = data1["nodes"]
  data["nodes_2"] = data2["nodes"]


  dis = math.sqrt((pose1[3] - pose2[3]) ** 2 + (pose1[11] - pose2[11]) ** 2)

  data["centers_1"] = data1["centers"]
  data["centers_2"] = data2["centers"]



  data["pcn_features_1"] = data1["bboxes"]
  data["pcn_features_2"] = data2["bboxes"]

  data["pcn_features_1"]=np.expand_dims(data["pcn_features_1"],axis=1)
  data["pcn_features_2"]=np.expand_dims(data["pcn_features_2"],axis=1)

  data["distance"] = dis


  return data


if __name__=="__main__":
  root_dir="/home/liudiyang/ms/refer/sgpr_imitate/data_preprocess/debug_bbox/graphs/"
  #
  path=[root_dir+"08/000000.npz",root_dir+"08/000005.npz"]
  data=process_pair_global_center(path)
  # print(data)
  # calib_filename = "/home/liudiyang/ms/dataset/semantic_kitti/data_odometry_calib/dataset/sequences/08/calib.txt"
  # lines = [line.rstrip() for line in open(calib_filename)]
  # print(len(lines))
  # velo_to_cam = np.array(lines[4].strip().split(" ")[1:], dtype=np.float64)
  # velo_to_cam=velo_to_cam.reshape(3,4)
