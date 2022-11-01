import torch
import torch.nn as nn
from torch import tensor
from sgpr_attention.src.models.components.layers_batch import SimpleAttentionModule,MultiheadAttentionModule,AttentionModule, TensorNetworkModule,DiffTensorNetworkModule,DiffDiffTensorNetworkModule
from sgpr_attention.src.models.components.dgcnn import get_graph_feature

class Edge_conv(nn.Module):
    def __init__(self, cfg, input_channel):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.K

        self.filters_dim=cfg.filters_dim
        self.layer_num = len(self.filters_dim)

        input_channel_list = self.filters_dim.copy()
        input_channel_list.pop()
        input_channel_list.insert(0,input_channel)

        output_channel_list = self.filters_dim

        self.dgcnn_conv_list = nn.ModuleList()
        for i in range(self.layer_num):
            self.dgcnn_conv_list.insert(len(self.dgcnn_conv_list),
                                        nn.Sequential(
                                            nn.Conv2d(2 * input_channel_list[i], output_channel_list[i], kernel_size=1),
                                            nn.BatchNorm2d(output_channel_list[i]),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Dropout(self.cfg.dropout)
                                        )
                                        )


    def edge_conv_forward(self, x, dgcnn_conv):

        x = get_graph_feature(x, k=self.k)  # Bx6xNxk
        x = dgcnn_conv(x) #[B,f,N,k]
        x1 = x.max(dim=-1, keepdim=False)[0]

        return x1

    def forward(self, x):

        for i in range(self.layer_num):
            x = self.edge_conv_forward(x, self.dgcnn_conv_list[i])

        return x

class SGPR_Baseline_Attention_Fusion(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.setup_layers()

    def calculate_bottleneck_features(self):
        """
        Deciding the shape of the bottleneck layer.
        """

        self.feature_count = self.cfg.tensor_neurons

    def setup_layers(self):
        """
        Creating the layers.
        """
        self.calculate_bottleneck_features()
        self.attention = SimpleAttentionModule(self.cfg)
        self.tensor_network = DiffDiffTensorNetworkModule(self.cfg)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.cfg.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.cfg.bottle_neck_neurons, 1)


        self.center_conv = Edge_conv(self.cfg, 3)
        self.sem_conv = Edge_conv(self.cfg, self.cfg.number_of_labels)

        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.cfg.filters_dim[-1] * 2,  # 3
                                                      self.cfg.filters_dim[-1], kernel_size=1, bias=False),
                                            nn.BatchNorm1d(self.cfg.filters_dim[-1]), nn.LeakyReLU(negative_slope=0.2))

        self.attention_fuse = nn.MultiheadAttention(64, 8, batch_first=True)
        self.layer_norm=nn.LayerNorm(64)
        self.pooling_layer=nn.AvgPool1d(2,2)
        
    def dgcnn_conv_pass(self, x):
        self.k = self.cfg.K


        xyz = x[:, self.cfg.geo_output_channels:self.cfg.geo_output_channels + 3, :]  # Bx3xN
        sem = x[:, self.cfg.geo_output_channels + 3:, :]  # BxfxN

        xyz = self.center_conv(xyz)
        sem = self.sem_conv(sem)  # [B,32,50]

        xyz = xyz.permute((0, 2, 1))  # [B,50,32]
        sem = sem.permute((0, 2, 1))

        fuse = torch.cat((xyz, sem), dim=2)  # [B,50,64]

        fuse = self.attention_fuse(fuse, fuse, fuse)[0]  # [B,50,64]

        # x=torch.cat((x,sem),dim=1)
        fuse=self.pooling_layer(fuse)
        # fuse = fuse.permute(0, 2, 1)
        x=fuse
        print(x.shape)
        # x = self.dgcnn_conv_end(fuse)
        # print(x.shape)
        # x = x.permute(0, 2, 1)  # [node_num, 32]

        return x

    def forward(self, data):
        features_1 = data["features_1"].cuda()
        features_2 = data["features_2"].cuda()  # [B,1024+3+12,N]
        # print("features shape",features_1.shape)
        B, _, N = features_1.shape

        # features B x (3+label_num) x node_num
        abstract_features_1 = self.dgcnn_conv_pass(features_1)  # node_num x feature_size(filters-3)
        abstract_features_2 = self.dgcnn_conv_pass(features_2)  # BXNXF

        # if self.cfg.histogram == True:
        #     hist = self.calculate_histogram(abstract_features_1,
        #                                     abstract_features_2.permute(0,2,1))

        pooled_features_1, attention_scores_1 = self.attention(abstract_features_1)  # bxfx1
        pooled_features_2, attention_scores_2 = self.attention(abstract_features_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = scores.permute(0, 2, 1)  # bx1xf
        #
        # if self.cfg.histogram == True:
        #     scores = torch.cat((scores, hist), dim=1).view(B,1, -1)
        # scores=self.dp(scores)


        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)

        return score, attention_scores_1, attention_scores_2