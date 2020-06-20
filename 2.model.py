import math
import torch
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(device)


class GraphConvolution(Module):
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)

        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.mm(adj, support)

        if self.bias is not None:
            return output + self.bias
        else:
            return output


class HyperGraphConvolution(Module):
    def __init__(self, in_channels, out_channels, bias=True):
        super(HyperGraphConvolution, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, self.out_channels))
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if bias:
            self.bias = Parameter(torch.FloatTensor(self.out_channels))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, theta):
        a = torch.mm(x, self.weight)
        output = torch.mm(theta, a)
        if self.bias is not None:
            return output + self.bias
        else:
            return output


class Machine(nn.Module):
    def __init__(self, node_number, dropout,
                 d_1=200, d_2=200, d_3=200,
                 ini_emb_mode='par',
                 embeddings_ae_or_one_hot=None):
        """
        ini_emb_mode=['par','ae','one_hot']
        d_1: ae,one-hot模式下无效，par模式下，self.ini_embeddings的维度，隐藏层的维度，也是输出层的维度
        d_2: ae，one-hot模式下，压缩的隐藏层维度
        d_3: ae,one-hot模式下，输出的维度
        """

        super(Machine, self).__init__()
        self.node_number = node_number
        if ini_emb_mode == 'par':
            self.d_1 = d_1  # 200
            self.d_2 = d_1  # 200
            self.d_3 = d_1  # 200
            self.ini_embeddings = torch.nn.Parameter(torch.Tensor(node_number, self.d_1))
            torch.nn.init.xavier_uniform_(self.ini_embeddings)
        elif (ini_emb_mode == 'ae') or (ini_emb_mode == 'one_hot'):

            self.d_1 = embeddings_ae_or_one_hot.shape[1]  # 8560 for one_hot or 200 for ae, maybe
            self.d_2 = d_2  # 200
            self.d_3 = d_3  # 200
            self.ini_embeddings = embeddings_ae_or_one_hot
        elif ini_emb_mode == 'manual_fea':
            self.d_1 = embeddings_ae_or_one_hot.shape[1]  # 36
            self.d_2 = self.d_1  # 36
            self.d_3 = d_3  # 16
            self.ini_embeddings = embeddings_ae_or_one_hot

        else:
            print('WONG INI_EMB_MODE!')

        self.dropout = dropout

        self.gcn_k = 5
        self.gcn1 = GraphConvolution(self.d_1, self.d_2)  # (200, 100)
        self.gcn2 = GraphConvolution(self.d_2 * self.gcn_k, self.d_3)
        self.hcn1 = HyperGraphConvolution(self.d_3, self.d_3)
        self.hcn2 = HyperGraphConvolution(self.d_3, self.d_3)

    def forward(self, adj=None, theta=None):

        x_list = []
        for k in range(self.gcn_k):
            x = F.relu(self.gcn1(self.ini_embeddings, adj))
            # x = F.dropout(x, self.dropout)
            x_list.append(x)

        x = torch.cat(x_list, dim=1)
        x = self.gcn2(x, adj)
        x = F.relu(self.hcn1(x, theta))
        x = F.dropout(x, self.dropout)
        self.embeddings_out = self.hcn2(x, theta)
        return self.embeddings_out

    def embedding_loss(self, embeddings, positive_links, negtive_links):

        left_p = embeddings[positive_links[:, 0]]
        right_p = embeddings[positive_links[:, 1]]
        dots_p = torch.sum(torch.mul(left_p, right_p), dim=1)
        positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
        left_n = embeddings[negtive_links[:, 0]]
        right_n = embeddings[negtive_links[:, 1]]
        dots_n = torch.sum(torch.mul(left_n, right_n), dim=1)
        negtive_loss = torch.mean(-1.0 * torch.log(1.01 - F.sigmoid(dots_n)))

        return positive_loss + negtive_loss

    def save_embeddings(self, data_name='facebook'):
        torch.save(self.embeddings_out, './' + data_name + '_before_' + self.mode + '.embeddings')


if __name__ == "__main__":

    """
    './{data_name}_{part_name}.adj' adj matrix for each partition of the dataset.
    './{data_name}_{part_name}.theta' calculated from hypergraph incident matirx(pyroch tensor)
    
    for a big graph, we make graph partition, each  partition is a part_name
    """

    epoches = 1000
    for data_name in ['network_1', 'network_2']:
        all_parts_name2index = pickle.load(open('./{}_all_parts.name2index'.format(data_name), 'rb'))
        part_number = len(all_parts_name2index.keys())
        for part_name in range(part_number):
            adj = torch.load('./{}_{}.adj'.format(data_name, part_name)).to(device)
            links_pd = pd.read_csv('./{}_{}.links'.format(data_name, part_name), header=None)
            links = torch.from_numpy(np.array(links_pd[[0, 1]]))
            links_target = torch.from_numpy(np.array(links_pd[2])).view(-1).to(device)
            positive_links = links[links_target == 1]
            negtive_links = links[links_target == 0]

            """
            './{data_name}_{part_name}.theta' calculated from hypergraph incident matirx(pyroch tensor)
            for a big graph, we make graph partition, each  partition is a part_name
            """
            theta_path = './{}_{}.theta'.format(data_name, part_name)
            theta = torch.load(theta_path).to(device)

            model = Machine(node_number=adj.shape[0],
                            dropout=0.001,
                            d_1=200, d_2=0, d_3=0,
                            ini_emb_mode='par').to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)

            model.train()

            for epoch in range(epoches):
                optimizer.zero_grad()

                embeddings = model(adj=adj, theta=theta)

                # print(embeddings)
                loss = model.embedding_loss(embeddings, positive_links, negtive_links)

                print("{} | part {}/{} | epoch {}/{} | loss: {:.4f}".format(data_name, part_name, part_number,
                                                                               epoch, epoches, loss.item()))
                loss.backward()
                optimizer.step()

            model.save_embeddings(data_name='{}_{}'.format(data_name, part_name))
