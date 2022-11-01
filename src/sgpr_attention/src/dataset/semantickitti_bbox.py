import torch
from torch.utils import data
from sgpr_attention.src.dataset.utils import load_paires, process_pair_bbox
from sgpr_attention.src.dataset.transforms import *
import os
from torch.utils.data.dataloader import default_collate
import numpy as np
from torchvision import transforms
import random

class SemanticKittiBbox(data.Dataset):
    def __init__(self, cfg, mode="train"):
        self.cfg = cfg
        self.pair_list_dir = cfg.pairs_dir
        self.graph_pairs_dir = cfg.graphs_dir

        self.graphs = []
        self.mode = mode

        self.point_cloud_transform = transforms.Compose([FlipPointCloud(),
                                                         RotatePointCloud(),
                                                         JitterPointCloud(),
                                                         RandomScalePointCloud(),
                                                         RotatePerturbationPointCloud(),
                                                         ShiftPointCloud(),
                                                         ])

        # self.feature_transform = transforms.Compose([FlipPointCloud(),
        #                                                  RotatePointCloud(),
        #                                                  JitterPointCloud(),
        #                                                 #  RandomScalePointCloud(),
        #                                                  RotatePerturbationPointCloud(),
        #                                                  ShiftPointCloud(),
        #                                                  ])
        self.point_cloud_transform = [FlipPointCloud(),
                                                         RotatePointCloud(),
                                                         JitterPointCloud(),
                                                         RandomScalePointCloud(),
                                                         RotatePerturbationPointCloud(),
                                                         ShiftPointCloud(),
                                                         ]

        self.shuffle = transforms.Compose([
            ShuffleFeature(),
        ])

        train_sequences = self.cfg.train_sequences
        eval_sequences = self.cfg.eval_sequences
        test_sequences = self.cfg.test_sequences

        if mode == "train":
            for sq in train_sequences:
                train_graphs = load_paires(os.path.join(self.cfg.pairs_dir, sq + ".txt"), self.graph_pairs_dir)
                self.graphs.extend(train_graphs)
        elif mode == "eval":
            for sq in eval_sequences:
                eval_graphs = load_paires(os.path.join(self.cfg.eval_pairs_dir, sq + ".txt"), self.graph_pairs_dir)
                self.graphs.extend(eval_graphs)
        else:
            for sq in test_sequences:
                test_graphs = load_paires(os.path.join(self.cfg.test_pairs_dir, sq + ".txt"), self.graph_pairs_dir)
                self.graphs.extend(test_graphs)
        # self.graphs=self.graphs[:10]
        # self.evaling_graphs=self.evaling_graphs[-10:]

        self.number_of_labels = self.cfg.number_of_labels
        self.global_labels = [i for i in range(self.number_of_labels)]  # 20
        self.global_labels = {val: index for index, val in enumerate(self.global_labels)}


    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):

        data = process_pair_bbox(self.graphs[idx])

        node_num_1 = len(data["nodes_1"])
        node_num_2 = len(data["nodes_2"])
        new_data = dict()

        if data["distance"] <= self.cfg.p_thresh:
            new_data["target"] = 1
        elif data["distance"] >= 20:
            new_data["target"] = 0.0
        else:
            new_data["target"] = 100
        if node_num_1 > self.cfg.node_num:
            sampled_index_1 = np.random.choice(node_num_1, self.cfg.node_num, replace=False)
            sampled_index_1.sort()
            data["nodes_1"] = np.array(data["nodes_1"])[sampled_index_1].tolist()
            data["pcn_features_1"] = np.array(data["pcn_features_1"])[sampled_index_1]
            data["centers_1"] = np.array(data["centers_1"])[sampled_index_1]
            # data["bbox_1"]=np.array(data["bbox"])[sampled_index_1]

        elif node_num_1 < self.cfg.node_num:
            data["nodes_1"] = np.concatenate(
                (np.array(data["nodes_1"]), -np.ones(self.cfg.node_num - node_num_1))).tolist()  # padding 0

            data["pcn_features_1"] = np.concatenate(
                (np.array(data["pcn_features_1"]),
                 np.zeros((self.cfg.node_num - node_num_1, 1, self.cfg.geo_output_channels))))  # padding 0
            data["centers_1"] = np.concatenate(
                (np.array(data["centers_1"]), np.zeros((self.cfg.node_num - node_num_1, 3))))  # padding 0
            # print(np.array(data["bbox_1"]).shape)np.array(data["bbox_1"])
            # data["bbox_1"]=np.concatenate(
            #     (np.array(data["bbox_1"]), np.zeros((self.cfg.node_num - node_num_1, 6))))

        if node_num_2 > self.cfg.node_num:
            sampled_index_2 = np.random.choice(node_num_2, self.cfg.node_num, replace=False)
            sampled_index_2.sort()
            data["nodes_2"] = np.array(data["nodes_2"])[sampled_index_2].tolist()
            data["pcn_features_2"] = np.array(data["pcn_features_2"])[sampled_index_2]  # node_num x 3
            data["centers_2"] = np.array(data["centers_2"])[sampled_index_2]  # node_num x 3

        elif node_num_2 < self.cfg.node_num:
            data["nodes_2"] = np.concatenate(
                (np.array(data["nodes_2"]), -np.ones(self.cfg.node_num - node_num_2))).tolist()
            data["pcn_features_2"] = np.concatenate(
                (np.array(data["pcn_features_2"]),
                 np.zeros((self.cfg.node_num - node_num_2, 1, self.cfg.geo_output_channels))))

            data["centers_2"] = np.concatenate(
                (np.array(data["centers_2"]), np.zeros((self.cfg.node_num - node_num_2, 3))))  # padding 0
            # data["bbox_2"]=np.concatenate(
            #     (np.array(data["bbox_2"]), np.zeros((self.cfg.node_num - node_num_1, 6))))
        # encoding the categories into one-hot code
        features_1 = np.expand_dims(np.array(
            [np.zeros(self.number_of_labels).tolist() if node == -1 else [
                1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()]
             for node in data["nodes_1"]]), axis=0)

        features_2 = np.expand_dims(np.array(
            [np.zeros(self.number_of_labels).tolist() if node == -1 else [
                1.0 if self.global_labels[node] == label_index else 0 for label_index in self.global_labels.values()]
             for node in data["nodes_2"]]), axis=0)

        # pcn-features
        pcn_features_1 = np.expand_dims(data["pcn_features_1"].reshape(self.cfg.node_num, -1), axis=0)  # [1,n,1024]
        pcn_features_2 = np.expand_dims(data["pcn_features_2"].reshape(self.cfg.node_num, -1), axis=0)
        # print(pcn_features_1.shape)
        # pcn_features_1=data["pcn_features_1"].reshape(self.cfg.node_num,-1)
        # pcn_features_2=data["pcn_features_2"].reshape(self.cfg.node_num,-1)
        xyz_1 = np.expand_dims(data["centers_1"], axis=0)#[1,N,3]
        xyz_2 = np.expand_dims(data["centers_2"], axis=0)

        # augment data
        if self.mode == "train":
            augment_number=np.random.randint(6,7)
            point_cloud_transform=random.choices(self.point_cloud_transform, k=augment_number)
            point_cloud_transform=transforms.Compose(point_cloud_transform)

            xyz_1 = point_cloud_transform(xyz_1)
            xyz_2 = point_cloud_transform(xyz_2)

            pcn_features_1=pcn_features_1.reshape((1,-1,3),order="F")
            pcn_features_2=pcn_features_2.reshape((1,-1,3),order="F")

            pcn_features_1=point_cloud_transform(pcn_features_1)
            pcn_features_2=point_cloud_transform(pcn_features_2)

            pcn_features_1=pcn_features_1.reshape(1,-1,6,order="F")#[1,N,6]
            pcn_features_2=pcn_features_2.reshape(1,-1,6,order="F")

            # pcn_features_1 = self.feature_transform(pcn_features_1)  # 1x50x1024
            # pcn_features_2 = self.feature_transform(pcn_features_2)


        total_feature_1 = np.concatenate((pcn_features_1, xyz_1, features_1), axis=2)  # .transpose(0, 2, 1)
        total_feature_2 = np.concatenate((pcn_features_2, xyz_2, features_2), axis=2)  # .transpose(0,2,1)

        # if self.mode == "train":
        #     total_feature_1 = self.shuffle(total_feature_1)
        #     total_feature_2 = self.shuffle(total_feature_2)

        total_feature_1 = total_feature_1.transpose(0, 2, 1)
        total_feature_2 = total_feature_2.transpose(0, 2, 1)

        new_data["features_1"] = np.squeeze(total_feature_1).astype(np.float)
        new_data["features_2"] = np.squeeze(total_feature_2).astype(np.float)  # [B,3M+12,N]

        new_data["features_1"] = torch.from_numpy(new_data["features_1"]).type(torch.FloatTensor)
        new_data["features_2"] = torch.from_numpy(new_data["features_2"]).type(torch.FloatTensor)
        # new_data["target"]=torch.from_numpy(new_data["target"]).type(torch.FloatTensor)
        return new_data

    def remove_ambiguity(self, batch):
        batch_size = len(batch)
        new_batch = []
        for i in range(batch_size):
            item = batch[i]
            if item["target"] != 100:
                new_batch.append(item)

        if len(new_batch) == 0:
            return None
        return default_collate(new_batch)

