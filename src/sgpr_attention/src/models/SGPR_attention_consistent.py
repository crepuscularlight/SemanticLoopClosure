import torch
import torch.nn as nn
from sgpr_attention.src.models.components.dgcnn import get_attention_feature,get_attention_feature_consistent
from sgpr_attention.src.models.components.layers_batch import MultiheadAttentionModule,AttentionModule, TensorNetworkModule,DiffDiffTensorNetworkModule

class GAT(nn.Module):
# get graph attention feature [B,N,k,2f]
    def __init__(self,input_channels,output_channels,k=10):
        super().__init__()
        self.k=k
        self.weight_matrix_list = nn.ModuleList()
        self.a_list=nn.ModuleList()
        self.head_num=int(output_channels/4)

        self.weights=torch.nn.Parameter(torch.Tensor(2*input_channels,4*self.head_num))
        self.a=torch.nn.Parameter(torch.Tensor(4,self.head_num))
        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.xavier_uniform_(self.a)


    def forward(self,x):
        B,N,k,f=x.shape
        output=torch.empty([B,N,0]).cuda()
        for i in range(self.head_num):
            hw=torch.matmul(x,self.weights[:,4*i:4*i+4])#[B,N,k,4]
            tmp=nn.functional.leaky_relu(hw,negative_slope=0.2)
            ahw=torch.matmul(tmp,self.a[:,i:i+1])#[B,N,k,1]
            weight=nn.functional.softmax(ahw,dim=2).permute(0,1,3,2)#[B,N,k,1]->[B,N,1,k]
            out=torch.matmul(weight,hw)#[B,N,1,4]
            out=torch.squeeze(out,dim=2)#[B,N,4]
            out=nn.functional.leaky_relu(out,negative_slope=0.2)
            output=torch.cat((output,out),dim=-1)

        output=output.permute(0,2,1)#[B,32,N]

        return output

class Refined_GAT(nn.Module):
    # input [B,N,k,2d]
    def __init__(self,input_channels,output_channels,k=10):
        super().__init__()
        self.k=k
        self.weight_matrix_list = nn.ModuleList()
        self.a_list=nn.ModuleList()
        self.head_num=int(output_channels/4)

        self.weights=torch.nn.Parameter(torch.Tensor(input_channels,input_channels))
        self.a=torch.nn.Parameter(torch.Tensor(input_channels,1))
        self.mlp1=torch.nn.Parameter(torch.Tensor(input_channels,output_channels//2))
        self.mlp2=torch.nn.Parameter(torch.Tensor(output_channels//2,output_channels//2))
        self.mlp3=torch.nn.Parameter(torch.Tensor(output_channels//2,output_channels))

        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.xavier_uniform_(self.a)
        torch.nn.init.xavier_uniform_(self.mlp1)
        torch.nn.init.xavier_uniform_(self.mlp2)
        torch.nn.init.xavier_uniform_(self.mlp3)


    def forward(self,x):
        B, N, k, f = x.shape
        h_origin=torch.squeeze(x[:,:,0,:f//2],dim=2)#[B,N,d]
        hw=torch.matmul(x[:,:,:,f//2:],self.weights) #[B,N,k,d]x[d,d]->[B,N,k,d]


        hwa=torch.matmul(hw,self.a) #[B,N,k,d]x[d,1]->[B,N,k,1]
        weight = nn.functional.softmax(hwa, dim=2).permute(0, 1, 3, 2)  # [B,N,k,1]->[B,N,1,k]
        out=torch.matmul(weight,hw)#[B,N,1,d]
        out = torch.squeeze(out, dim=2)  # [B,N,d]
        out=out+h_origin

        out=torch.matmul(out,self.mlp1) #[B,N,32]
        out=torch.matmul(out,self.mlp2) #[B,N,32]
        out=torch.matmul(out,self.mlp3) #[B,N,32]

        out = out.permute(0, 2, 1)  # [B,32,N]

        return out

class res_GAT(nn.Module):
# get graph attention feature [B,N,k,2f]
    def __init__(self,input_channels,output_channels,k=10):
        super().__init__()
        self.k=k
        self.weight_matrix_list = nn.ModuleList()
        self.a_list=nn.ModuleList()
        self.head_num=int(output_channels/4)

        self.weights=torch.nn.Parameter(torch.Tensor(2*input_channels,4*self.head_num))
        self.a=torch.nn.Parameter(torch.Tensor(4,self.head_num))
        self.mlp1=torch.nn.Parameter(torch.Tensor(input_channels,output_channels))
        self.mlp2=torch.nn.Parameter(torch.Tensor(output_channels,output_channels))

        torch.nn.init.xavier_uniform_(self.weights)
        torch.nn.init.xavier_uniform_(self.a)
        torch.nn.init.xavier_uniform_(self.mlp1)
        torch.nn.init.xavier_uniform_(self.mlp2)


    def forward(self,x):
        B,N,k,f=x.shape
        output=torch.empty([B,N,0]).cuda()
        h_origin=torch.squeeze(x[:,:,0,:f//2],dim=2)
        for i in range(self.head_num):
            
            hw=torch.matmul(x,self.weights[:,4*i:4*i+4])#[B,N,k,4]
            tmp=nn.functional.leaky_relu(hw,negative_slope=0.2)
          
            ahw=torch.matmul(tmp,self.a[:,i:i+1])#[B,N,k,1]
            weight=nn.functional.softmax(ahw,dim=2).permute(0,1,3,2)#[B,N,k,1]->[B,N,1,k]
            out=torch.matmul(weight,hw)#[B,N,1,4]
            out=torch.squeeze(out,dim=2)#[B,N,4]
            output=torch.cat((output,out),dim=-1)

        h_origin=torch.matmul(h_origin,self.mlp1) #[B,N,32]
        h_origin=torch.matmul(h_origin,self.mlp2) #[B,N,32]

        output=output+h_origin
        # output=torch.matmul(output,self.mlp2)
        output=output.permute(0,2,1)#[B,32,N]

        return output

class Graph_Attention_Consistent(nn.Module):
    def __init__(self, cfg, input_channel_cen,input_channel_sem):
        super().__init__()
        self.cfg = cfg
        self.k = cfg.K

        self.filters_dim=cfg.filters_dim
        self.layer_num = len(self.filters_dim)

        self.input_channel_list_cen = self.filters_dim.copy()
        self.input_channel_list_cen.pop()
        self.input_channel_list_cen.insert(0,input_channel_cen)

        self.input_channel_list_sem = self.filters_dim.copy()
        self.input_channel_list_sem.pop()
        self.input_channel_list_sem.insert(0,input_channel_sem)

        output_channel_list = self.filters_dim

        self.attention_conv_list_cen = nn.ModuleList()
        self.attention_conv_list_sem = nn.ModuleList()


        for i in range(self.layer_num):
            self.attention_conv_list_cen.insert(len(self.attention_conv_list_cen),
                                            nn.Sequential(
                                            res_GAT(self.input_channel_list_cen[i],output_channel_list[i]),
                                            nn.BatchNorm1d(output_channel_list[i]),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Dropout(self.cfg.dropout)
                                            )
                                        )
            self.attention_conv_list_sem.insert(len(self.attention_conv_list_sem),
                                            nn.Sequential(
                                            res_GAT(self.input_channel_list_sem[i],output_channel_list[i]),
                                            nn.BatchNorm1d(output_channel_list[i]),
                                            nn.LeakyReLU(negative_slope=0.2),
                                            nn.Dropout(self.cfg.dropout)
                                            )
                                        )



    def graph_attention_forward(self, x,attention_cen,attention_sem,xyz_len,sem_len):

        x = get_attention_feature_consistent(x, k=self.k,xyz=True,xyz_len=xyz_len)  # BxNxkx2d
        cen=torch.cat((x[:,:,:,:xyz_len],x[:,:,:,xyz_len+sem_len:xyz_len+sem_len+xyz_len]),dim=-1)
        sem=torch.cat((x[:,:,:,xyz_len:xyz_len+sem_len],x[:,:,:,2*xyz_len+sem_len:]),dim=-1)

        cen = attention_cen(cen) #[B,f,N,k]


        sem= attention_sem(sem) #[B,f,N,k]
  

        return torch.cat((cen,sem),dim=1)

    def forward(self, x):

        for i in range(self.layer_num):
            x = self.graph_attention_forward(x, self.attention_conv_list_cen[i],self.attention_conv_list_sem[i],
            self.input_channel_list_cen[i],self.input_channel_list_sem[i])

        return x

class SGPR_Attention_Consistent(nn.Module):
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
        self.attention = AttentionModule(self.cfg)
        self.tensor_network = DiffDiffTensorNetworkModule(self.cfg)
        self.fully_connected_first = torch.nn.Linear(self.feature_count, self.cfg.bottle_neck_neurons)
        self.scoring_layer = torch.nn.Linear(self.cfg.bottle_neck_neurons, 1)


        self.center_sem_conv = Graph_Attention_Consistent(self.cfg, 3,self.cfg.number_of_labels)

        self.dgcnn_conv_end = nn.Sequential(nn.Conv1d(self.cfg.filters_dim[-1] * 2,  # 3
                                                      self.cfg.filters_dim[-1], kernel_size=1, bias=False),
                                            nn.BatchNorm1d(self.cfg.filters_dim[-1]), nn.LeakyReLU(negative_slope=0.2))



    def dgcnn_conv_pass(self, x):
        self.k = self.cfg.K
        xyz_sem = x[:, self.cfg.geo_output_channels:, :]  # Bx3xN

        xyz_sem=self.center_sem_conv(xyz_sem)


        # geo=torch.unsqueeze(geo,dim=1)
        # xyz=torch.unsqueeze(xyz,dim=1)
        # sem=torch.unsqueeze(sem,dim=1)

        # print(geo.shape)
        # x = torch.cat((geo,xyz), dim=1)
        x = xyz_sem
        # print("x1 shape",x.shape)
        # x=self.fusion_conv(x)

        # x=self.fusion_conv_1(x)
        # print("x shape",x.shape)
        # x=torch.squeeze(x,dim=1)
        # x=torch.cat((x,sem),dim=1)
        x = self.dgcnn_conv_end(x)
        # print(x.shape)
        x = x.permute(0, 2, 1)  # [node_num, 32]

        return x

    def forward(self, data):
        features_1 = data["features_1"].cuda()
        features_2 = data["features_2"].cuda()  # [B,1024+3+12,N]
        # print("features shape",features_1.shape)
        B, _, N = features_1.shape

        # features B x (3+label_num) x node_num
        abstract_features_1 = self.dgcnn_conv_pass(features_1)  # node_num x feature_size(filters-3)
        abstract_features_2 = self.dgcnn_conv_pass(features_2)  # BXNXF


        pooled_features_1, attention_scores_1 = self.attention(abstract_features_1)  # bxfx1
        pooled_features_2, attention_scores_2 = self.attention(abstract_features_2)

        scores = self.tensor_network(pooled_features_1, pooled_features_2)
        scores = scores.permute(0, 2, 1)  # bx1xf

        scores = torch.nn.functional.relu(self.fully_connected_first(scores))
        score = torch.sigmoid(self.scoring_layer(scores)).reshape(-1)

        return score, attention_scores_1, attention_scores_2
