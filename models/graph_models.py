import torch
import argparse
import csv
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, RGATConv, GAT, GATv2Conv, TransformerConv
from torch_geometric.data import Data, HeteroData
from torch_geometric.data import Batch
from torch_geometric.utils import subgraph, add_self_loops, to_dense_batch
class RGAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, heads, num_relations, edge_dim, dropout, is_full=False):
        """
        初始化关系图注意力网络（RGAT）层。

        参数:
        :param in_channels: 输入通道数。
        :param hidden_channels: 隐藏通道数。
        :param out_channels: 输出通道数。
        :param heads: 注意力头的数量。
        :param num_relations: 关系类型数量。
        :param edge_dim: 边特征的维度。
        :param dropout: Dropout比率。
        :param is_full: 布尔值，指示是否使用全注意力机制。
        """
        super(RGAT, self).__init__()
        
        # 初始化第一层变换器卷积层
        self.cn1 = TransformerConv(in_channels, hidden_channels, edge_dim=edge_dim, dropout=dropout, heads=heads, bias=False)
        self.is_full = is_full
        
        # 初始化边嵌入层，将关系索引转换为边特征
        self.edge_emb_layer = torch.nn.Embedding(num_relations, edge_dim)
        
        # 检查隐藏通道乘以注意力头数是否等于输出通道数
        # 如果不等，添加一个投影层来调整注意力机制的输出到期望的维度
        if hidden_channels*heads != out_channels:
            self.project_layer = torch.nn.Linear(hidden_channels*heads, out_channels, bias=False)
    def forward(self, x, edge_index, edge_type, return_attention_weights=False):
        """
        前向传播函数，用于计算图神经网络的一个迭代步骤。
        
        参数:
        - x: 图节点的特征矩阵。
        - edge_index: 图的边索引矩阵，用于定义节点之间的连接关系。
        - edge_type: 边的类型矩阵，用于定义不同边的属性。
        - return_attention_weights: 是否返回注意力权重，默认为False。
        
        返回:
        - 如果 `return_attention_weights` 为True，则返回节点特征矩阵x和注意力权重alpha、beta。
        - 如果 `return_attention_weights` 为False，则只返回节点特征矩阵x。
        """
        # 根据边的类型查找对应的边属性
        edge_attr = self.edge_emb_layer(edge_type)
        
        if return_attention_weights:
            # 使用第一个卷积层进行计算，并返回注意力权重
            x, (alpha, beta) = self.cn1(x, edge_index, edge_attr=edge_attr, return_attention_weights=return_attention_weights)
        else:
            # 使用第一个卷积层进行计算，不返回注意力权重
            x = self.cn1(x, edge_index, edge_attr=edge_attr)
            # 如果存在投影层，则对输出进行投影
            if hasattr(self, 'project_layer'):
                x = self.project_layer(x)
        
        if return_attention_weights:
            # 如果需要返回注意力权重，则返回x、alpha和beta
            return x, alpha, beta
        else:
            # 否则，只返回x
            return x
import copy
class KGCodeReassign(torch.nn.Module):
    """
    代码重分配类，使用图注意力网络对代码进行重分配。
    
    初始化方法初始化了类实例，包括图模型、代码映射和图的边缘信息。
    
    参数:
    - args: 包含模型参数的字典。
    - edges_dict: 边缘字典，描述代码元素之间的关系。
    - c2ind: 常量到索引的映射字典。
    - cm2ind: 模型常量到索引的映射字典。
    """
    def __init__(self, args, edges_dict, c2ind, cm2ind):
        # 调用超类的初始化方法
        super(KGCodeReassign, self).__init__()
        # 初始化关系图注意力网络模型
        # self.GATmodel = GAT(args["num_features"], args["hidden_channels"], args["embedding_dim"],
        #                     args["heads"]).to(args.device)
        self.RGATmodel = RGAT(args['attention_dim'], args['attention_dim']//args['use_multihead'], args['attention_dim'], args['use_multihead'],
                              num_relations=11, edge_dim=args['edge_dim'], dropout=args['rep_dropout']/2, is_full=len(c2ind)>50)
        # 初始化常量代码数量
        self.original_code_num = len(c2ind)
        # 深拷贝常量和模型常量的索引映射，以保持原始数据的不变性
        self.c2ind = copy.deepcopy(c2ind)
        self.cm2ind = copy.deepcopy(cm2ind)
        # 初始化模型常量列表
        self.mcodes = []
        # 初始化边缘列表和类型列表
        edges = [[],[]]
        edges_type = []
        # 遍历边缘字典，填充边缘和类型信息
        for edge_pair in edges_dict.keys():
            if edges_dict[edge_pair] != 0:
                edges[0].append(self.cm2ind[edge_pair[0]] + self.original_code_num)
                edges[1].append(self.c2ind[edge_pair[1]])
            elif edges_dict[edge_pair] == 0:
                edges[0].append(self.c2ind[edge_pair[0]])
                edges[1].append(self.c2ind[edge_pair[1]])
            edges_type.append(edges_dict[edge_pair])
        # 将模型常量索引转换为参数，不参与梯度更新
        self.mcodes = torch.nn.Parameter(torch.arange(0, len(cm2ind))+len(c2ind), requires_grad=False)
        # 将边缘信息转换为参数，不参与梯度更新
        self.edges = torch.nn.Parameter(torch.LongTensor(edges), requires_grad=False)
        self.edges_type = torch.nn.Parameter(torch.LongTensor(edges_type), requires_grad=False)
        # 构建索引到常量的反向映射
        self.ind2c = {v: k for k, v in self.c2ind.items()}
        self.ind2mc = {v+self.original_code_num: k for k, v in self.cm2ind.items()}
        self.ind2mc.update(self.ind2c)
        #self.ind2c.update(self.ind2mc)

    def forward(self, code_embeddings, mcode_embeddings, indices, return_attention_weights=False):
        """
        前向传播函数，用于处理代码嵌入和多代码嵌入，通过图注意力网络（GAT）得到图嵌入。
        
        参数:
        - code_embeddings: 代码嵌入张量，形状为[B, C, E]或[B, E]，B为批次大小，C为代码类别数量，E为嵌入维度。
        - mcode_embeddings: 多代码嵌入张量，形状与code_embeddings相同。
        - indices: 需要处理的代码索引，如果为None，则处理所有代码。
        - return_attention_weights: 是否返回注意力权重，用于解释模型的决策。
        
        返回:
        - graph_embeddings: 图嵌入张量，形状与输入嵌入相同。
        - attentions_rs: 关键代码和其注意力权重的字典列表，仅当return_attention_weights为True时返回。
        """
        #[B, C, E]
        batch_data = []
        edges_reals = []
        # 处理code_embeddings维度为2的情况，即单个代码嵌入
        if len(code_embeddings.shape) == 2:
            if indices is not None:
                # 根据indices和多代码嵌入构建子图
                edges, edges_type = subgraph(torch.cat([indices, self.mcodes], dim=0), self.edges, self.edges_type, relabel_nodes=True)
                if return_attention_weights:
                    edges_real, _ = subgraph(torch.cat([indices, self.mcodes], dim=0), self.edges, self.edges_type, relabel_nodes=False)
                    edges_reals.append(edges_real)
            else:
                edges = self.edges
                edges_type = self.edges_type
            # 合并代码嵌入和多代码嵌入
            topk_code_embedding = torch.cat([code_embeddings, mcode_embeddings], dim=0)
            batch_data.append(Data(x=topk_code_embedding, edge_index=edges, edge_type=edges_type))
        # 处理code_embeddings维度为3的情况，即多个代码嵌入
        else:
            for ind in range(len(code_embeddings)):
                if indices is not None:
                    edges, edges_type = subgraph(torch.cat([indices, self.mcodes], dim=0), self.edges, self.edges_type, relabel_nodes=True)
                    if return_attention_weights:
                        edges_real, _ = subgraph(torch.cat([indices, self.mcodes], dim=0), self.edges, self.edges_type, relabel_nodes=False)
                        edges_reals.append(edges_real)
                else:
                    edges = self.edges
                    edges_type = self.edges_type
                # 合并单个代码嵌入和多代码嵌入
                topk_code_embedding = torch.cat([code_embeddings[ind], mcode_embeddings[ind]], dim=0)
                batch_data.append(Data(x=topk_code_embedding, edge_index=edges, edge_type=edges_type))
        # 创建批次数据对象
        batch = Batch.from_data_list(batch_data)
        # 根据是否返回注意力权重进行模型推理
        if return_attention_weights:
            graph_embeddings_, edge_index_used, attention_weights = self.RGATmodel(batch.x, batch.edge_index, edge_type=batch.edge_type, return_attention_weights=return_attention_weights)
            edge_index_used = edge_index_used.view(2, -1, edges.shape[1])
            attention_weights = attention_weights.mean(dim=1)
            attention_weights = attention_weights.view(-1, edges.shape[1])
            delta = edge_index_used.shape[-1]
            attentions_rs = []
            for bid in range(code_embeddings.shape[0]):
                if indices is None:
                    current_edge_index = edge_index_used[:, bid, :] - delta*bid
                else:
                    current_edge_index = edges_reals[bid]
                current_attention_weights = attention_weights[bid, :]
                attentions_dict = {}
                for code_key in self.c2ind.keys():
                    code_id = self.c2ind[code_key]
                    index = (current_edge_index[1] == code_id)
                    if index.sum() > 0:
                        attentions_of_code = current_attention_weights[index]
                        source_code_id = current_edge_index[0][index]
                        source_code_names = [self.ind2mc[int(i)] for i in source_code_id]
                        attentions_of_code = attentions_of_code.cpu().detach().numpy()
                        results = list(zip(source_code_names, attentions_of_code))
                        results = sorted(results, key=lambda x: x[1], reverse=True)[0:10]
                        if attentions_of_code.max() > 0.0015:
                            print(code_key, results)
                        attentions_dict[code_key] = results
                attentions_rs.append(attentions_dict)
        else:
            graph_embeddings_ = self.RGATmodel(batch.x, batch.edge_index, edge_type=batch.edge_type)
        # 根据code_embeddings的维度调整返回的图嵌入形状
        if len(code_embeddings.shape) == 3:
            graph_embeddings = to_dense_batch(graph_embeddings_, batch.batch)[0]
            graph_embeddings = graph_embeddings[:, 0:code_embeddings.shape[1]]
        else:
            graph_embeddings = graph_embeddings_[0:code_embeddings.shape[0]]
        # 返回图嵌入，如果需要，还包括注意力权重
        if return_attention_weights:
            return graph_embeddings, attentions_rs
        else:
            return graph_embeddings