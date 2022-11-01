import torch
class SimpleAttentionModule(torch.nn.Module):

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last = args.filters_dim[-1]
        self.setup_weights()
        self.init_parameters()


    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.filters_last, self.filters_last))

        self.a=torch.nn.Parameter(torch.Tensor(self.filters_last, 8))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.a)


    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """

        batch_size = embedding.shape[0]

        ew = torch.matmul(embedding, self.weight_matrix)  #[B,N,f] x [f,f] ->[B,N,f]
        
        aew=torch.matmul(ew,self.a) #[B,N,f]x[f,8]->[B,N,8]
        aew=torch.nn.functional.leaky_relu(aew,negative_slope=0.2)
        aew_avg=torch.mean(aew,dim=2)
        aew_avg=torch.unsqueeze(aew_avg,dim=2)
        weights=torch.nn.functional.softmax(aew_avg,dim=1) #[B,N,1]

        representation = torch.matmul(embedding.permute(0, 2, 1), weights)  # [B,f,N]x[B,N,1] -> [B,f,1]

        return representation, None
# class MultiheadAttentionModule(torch.nn.Module):
#     """
#     SimGNN Attention Module to make a pass on graph.
#     """

#     def __init__(self, args):
#         """
#         :param args: Arguments object.
#         """
#         super().__init__()
#         self.args = args
#         self.filters_last = args.filters_dim[-1]

#         self.attention_layer=torch.nn.MultiheadAttention(self.filters_last,8,batch_first=True)
#     def forward(self, embedding):
#         """
#         Making a forward propagation pass to create a graph level representation.
#         :param embedding: Result of the GCN. [B,N,32]
#         :return representation: A graph level representation vector.
#         """

#         batch_size = embedding.shape[0]
#         # global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1)  # [B,N,32] [32,32]->[B,N,32] ->[B,32]
#         global_context=self.attention_layer(embedding,embedding,embedding)[0]
#         global_context=torch.mean(global_context,dim=1)

#         transformed_global = torch.tanh(global_context)  # [B,32]
#         # transformed_global = torch.nn.functional.tanh(global_context)  # [B,32]
#         sigmoid_scores = torch.sigmoid(torch.matmul(embedding, transformed_global.view(batch_size, -1,1))) #[B,N,32]x[B,32,1]->[B,N,1]
#         representation = torch.matmul(embedding.permute(0, 2, 1), sigmoid_scores)  # [B,32,N] x[B,N,1] -> [B,32,1]

#         return representation, sigmoid_scores
class MultiheadAttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last = args.filters_dim[-1]

        self.attention_layer=torch.nn.MultiheadAttention(self.filters_last,8,batch_first=True,dropout=0.05)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN. [B,N,32]
        :return representation: A graph level representation vector.
        """

        batch_size = embedding.shape[0]
        # embedding=embedding.permute(0,2,1)
        # global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1)  # [B,N,32] [32,32]->[B,N,32] ->[B,32]
        global_context=self.attention_layer(embedding,embedding,embedding)[0] #[B,50,32]
        global_context=torch.mean(global_context,dim=1)

        # transformed_global = global_context  # [B,32]
        transformed_global = torch.nn.functional.tanh(global_context)  # [B,32]
        # embedding=embedding.permute(0,2,1)
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding, transformed_global.view(batch_size, -1,1))) #[B,N,32]x[B,32,1]->[B,N,1]
        representation = torch.matmul(embedding.permute(0, 2, 1), sigmoid_scores)  # [B,32,N] x[B,N,1] -> [B,32,1]ights)  # [B,f,N]x[B,N,1] -> [B,f,1]

        return representation, None

class MeanMultiheadAttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last = args.filters_dim[-1]

        self.attention_layer=torch.nn.MultiheadAttention(self.filters_last,8,batch_first=True)
        self.a=torch.nn.Parameter(torch.Tensor(2, 1))
        torch.nn.init.xavier_uniform_(self.a)
    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN. [B,N,32]
        :return representation: A graph level representation vector.
        """

        batch_size = embedding.shape[0]
        # embedding=embedding.permute(0,2,1)
        # global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1)  # [B,N,32] [32,32]->[B,N,32] ->[B,32]
        global_context=self.attention_layer(embedding,embedding,embedding)[0] #[B,50,32]
        global_context=torch.mean(global_context,dim=1)
        global_context=torch.unsqueeze(global_context,dim=2)#[B,32,1]

        global_context = torch.nn.functional.gelu(global_context)  # [B,32,1]

        mean_context= torch.mean(embedding,dim=1,keepdim=True)#[B,1,32]
        mean_context=mean_context.permute(0,2,1)#[B,32,1]

        representation=torch.cat((global_context,mean_context),dim=2) #[B,32,2]

        representation=torch.matmul(representation,self.a)
        # # transformed_global = torch.nn.functional.tanh(global_context)  # [B,32]
        # # embedding=embedding.permute(0,2,1)
        # sigmoid_scores = torch.sigmoid(torch.matmul(embedding, transformed_global.view(batch_size, -1,1))) #[B,N,32]x[B,32,1]->[B,N,1]
        # representation = torch.matmul(embedding.permute(0, 2, 1), sigmoid_scores)  # [B,32,N] x[B,N,1] -> [B,32,1]  # [B,f,N]x[B,N,1] -> [B,f,1]
        # print(global_context.shape)
        return representation, None

class AttentionModule(torch.nn.Module):
    """
    SimGNN Attention Module to make a pass on graph.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last =args.filters_dim[-1]
        self.setup_weights()
        self.init_parameters()


    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(torch.Tensor(self.filters_last, self.filters_last))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)

    def forward(self, embedding):
        """
        Making a forward propagation pass to create a graph level representation.
        :param embedding: Result of the GCN.
        :return representation: A graph level representation vector.
        """

        batch_size = embedding.shape[0]
        global_context = torch.mean(torch.matmul(embedding, self.weight_matrix), dim=1)  # 0 # nxf -> f  bxnxf->bxf
        transformed_global = torch.tanh(global_context)  # f  bxf
        sigmoid_scores = torch.sigmoid(torch.matmul(embedding, transformed_global.view(batch_size, -1,
                                                                                       1)))  # weights      nxf fx1  bxnxf bxfx1 bxnx1
        representation = torch.matmul(embedding.permute(0, 2, 1), sigmoid_scores)  # bxnxf bxfxn bxnx1 bxfx1

        return representation, sigmoid_scores


class TensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last=args.filters_dim[-1]
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.filters_last, self.filters_last, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2 * self.filters_last))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.    bxfx1
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = embedding_1.shape[0]
        scoring = torch.matmul(embedding_1.permute(0, 2, 1), self.weight_matrix.view(self.filters_last, -1)).view(
            batch_size, self.filters_last, self.args.tensor_neurons)
        scoring = torch.matmul(scoring.permute(0, 2, 1), embedding_2)  # bxfx1
        combined_representation = torch.cat((embedding_1, embedding_2), dim=1)  # bx2fx1
        block_scoring = torch.matmul(self.weight_matrix_block, combined_representation)  # bxtensor_neuronsx1
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias)
        return scores

class DiffTensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last=2*args.filters_dim[-1]
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.filters_last, self.filters_last, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2 * self.filters_last))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))
        self.difference_matrix=torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,self.filters_last))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.difference_matrix)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.    bxfx1
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = embedding_1.shape[0]
        scoring = torch.matmul(embedding_1.permute(0, 2, 1), self.weight_matrix.view(self.filters_last, -1)).view(
            batch_size, self.filters_last, self.args.tensor_neurons) #[B,1,32] x [32,32x16]-> [B,1,32x16]-> [B,32,16]
        scoring = torch.matmul(scoring.permute(0, 2, 1), embedding_2)  # [B,16,32]x[B,32,1] -> [B,16,1]
        combined_representation = torch.cat((embedding_1, embedding_2), dim=1)  # bx2fx1
        block_scoring = torch.matmul(self.weight_matrix_block, combined_representation)  # [16,64] x[B,64,1]->[B,16,1]
        difference=torch.abs(embedding_1-embedding_2)
        difference=torch.matmul(self.difference_matrix,difference) #[16,32]x[B,32,1] -> [B,16,1]
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias+difference) #[B,16,1]+[B,16,1]+[16,1]->[B,16,1]
        return scores
class DiffDiffTensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last=args.filters_dim[-1]
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.filters_last, self.filters_last, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2 * self.filters_last))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))
        self.difference_matrix=torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,self.filters_last))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.difference_matrix)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.    bxfx1
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = embedding_1.shape[0]
        scoring = torch.matmul((embedding_1-embedding_2).permute(0, 2, 1), self.weight_matrix.view(self.filters_last, -1)).view(
            batch_size, self.filters_last, self.args.tensor_neurons) #[B,1,32] x [32,32x16]-> [B,1,32x16]-> [B,32,16]
        scoring = torch.matmul(scoring.permute(0, 2, 1), (embedding_1-embedding_2))  # [B,16,32]x[B,32,1] -> [B,16,1]
        combined_representation = torch.cat((embedding_1, embedding_2), dim=1)  # bx2fx1
        block_scoring = torch.matmul(self.weight_matrix_block, combined_representation)  # [16,64] x[B,64,1]->[B,16,1]
        difference=torch.abs(embedding_1-embedding_2)
        difference=torch.matmul(self.difference_matrix,difference) #[16,32]x[B,32,1] -> [B,16,1]
        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias+difference) #[B,16,1]+[B,16,1]+[16,1]->[B,16,1]
        return scores

class SumDiffDiffTensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last=args.filters_dim[-1]
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.filters_last, self.filters_last, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2 * self.filters_last))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))
        self.difference_matrix=torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,self.filters_last))
        self.addition_matrix=torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,self.filters_last))

    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.difference_matrix)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.    bxfx1
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = embedding_1.shape[0]

        scoring = torch.matmul((embedding_1-embedding_2).permute(0, 2, 1), self.weight_matrix.view(self.filters_last, -1)).view(
            batch_size, self.filters_last, self.args.tensor_neurons) #[B,1,32] x [32,32x16]-> [B,1,32x16]-> [B,32,16]
        scoring = torch.matmul(scoring.permute(0, 2, 1), (embedding_1-embedding_2))  # [B,16,32]x[B,32,1] -> [B,16,1]

        combined_representation = torch.cat((embedding_1, embedding_2), dim=1)  # bx2fx1
        block_scoring = torch.matmul(self.weight_matrix_block, combined_representation)  # [16,64] x[B,64,1]->[B,16,1]

        difference=torch.abs(embedding_1-embedding_2)
        difference=torch.matmul(self.difference_matrix,difference) #[16,32]x[B,32,1] -> [B,16,1]

        addition=embedding_1+embedding_2
        addition=torch.matmul(self.addition_matrix,addition) #[16,32]x[B,32,1] -> [B,16,1]

        scores = torch.nn.functional.relu(scoring + block_scoring + self.bias+difference+addition) #[B,16,1]+[B,16,1]+[16,1]->[B,16,1]
        return scores

class SymDiffDiffTensorNetworkModule(torch.nn.Module):
    """
    SimGNN Tensor Network module to calculate similarity vector.
    """

    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super().__init__()
        self.args = args
        self.filters_last=args.filters_dim[-1]
        self.setup_weights()
        self.init_parameters()

    def setup_weights(self):
        """
        Defining weights.
        """
        self.weight_matrix = torch.nn.Parameter(
            torch.Tensor(self.filters_last, self.filters_last, self.args.tensor_neurons))
        self.weight_matrix_block = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 2 * self.filters_last))
        self.bias = torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, 1))
        self.difference_matrix=torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons,self.filters_last))
        self.sym_matrix=torch.nn.Parameter(torch.Tensor(self.args.tensor_neurons, self.filters_last))


    def init_parameters(self):
        """
        Initializing weights.
        """
        torch.nn.init.xavier_uniform_(self.weight_matrix)
        torch.nn.init.xavier_uniform_(self.weight_matrix_block)
        torch.nn.init.xavier_uniform_(self.bias)
        torch.nn.init.xavier_uniform_(self.difference_matrix)
        torch.nn.init.xavier_uniform_(self.sym_matrix)

    def forward(self, embedding_1, embedding_2):
        """
        Making a forward propagation pass to create a similarity vector.
        :param embedding_1: Result of the 1st embedding after attention.    bxfx1
        :param embedding_2: Result of the 2nd embedding after attention.
        :return scores: A similarity score vector.
        """
        batch_size = embedding_1.shape[0]
        scoring = torch.matmul((embedding_1-embedding_2).permute(0, 2, 1), self.weight_matrix.view(self.filters_last, -1)).view(
            batch_size, self.filters_last, self.args.tensor_neurons) #[B,1,32] x [32,32x16]-> [B,1,32x16]-> [B,32,16]
        scoring = torch.matmul(scoring.permute(0, 2, 1), (embedding_1-embedding_2))  # [B,16,32]x[B,32,1] -> [B,16,1]
        
        combined_representation = torch.cat((embedding_1, embedding_2), dim=1)  # bx2fx1
        block_scoring = torch.matmul(self.weight_matrix_block, combined_representation)  # [16,64] x[B,64,1]->[B,16,1]
        
        difference=torch.abs(embedding_1-embedding_2) #[B,32,1]
        difference=torch.matmul(self.difference_matrix,difference) #[16,32]x[B,32,1] -> [B,16,1]

        sym_difference=torch.matmul(embedding_1,embedding_2.permute(0,2,1)) #[B,32,32]
        sym_difference=torch.matmul(self.sym_matrix,sym_difference) #[B,16,32]x[B,32,32] -> [B,16,32]
        sym_difference=torch.mean(sym_difference,dim=2,keepdim=True)
        
        tmp=scoring + block_scoring + self.bias
        # tmp=torch.cat((tmp,difference),dim=1)
        scores = torch.nn.functional.relu(difference) #[B,16,1]+[B,16,1]+[16,1]->[B,16,1]

        return scores