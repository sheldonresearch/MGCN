import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class Matching(nn.Module):
    def __init__(self, d_embedding_before):
        super(Matching, self).__init__()
        self.fc1 = nn.Linear(d_embedding_before, d_embedding_before)
        self.fc2 = nn.Linear(d_embedding_before, d_embedding_before)

    def forward(self, embedding_1, embedding_2, observed_anchors_p):

        self.embedding_1_after = embedding_1
        # self.embedding_1_after = F.dropout(self.embedding_1_after, 0.5) # matching 的时候千万不要加dropout！
        # self.embedding_1_after = self.fc2(self.embedding_1_after)
        self.embedding_2_after = self.fc1(embedding_2)
        # print(self.embedding_1_after)
        x_1_p = self.embedding_1_after[observed_anchors_p[:, 0]]

        x_2_p = self.embedding_2_after[observed_anchors_p[:, 1]]

        dots_p = torch.sum(torch.mul(x_1_p, x_2_p), dim=1)
        positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
        return positive_loss

        #dis_p = torch.mean(F.pairwise_distance(x_1_p, x_2_p))
        #return dis_p

    def save_embeddings(self, mode='gcn_only', embedding_1_name='network_1', embedding_2_name='network_2'):
        torch.save(self.embedding_1_after, './' + embedding_1_name + '_after_' + mode + '.embeddings')
        torch.save(self.embedding_2_after, './' + embedding_2_name + '_after_' + mode + '.embeddings')


if __name__ == "__main__":

    epoches = 5000

    print('网络内matching...')
    for data_name in ['network_1']:
        """
            if network_name == 'facebook':
        network_alias = 'network_1'
        shared_number = 1056
    elif network_name == 'twitter':
        network_alias = 'network_2'
        shared_number = 1138
        """
        shared_number = 1056
        anchors_list = [[i, i] for i in range(shared_number)]
        anchors_p = torch.from_numpy(np.array(anchors_list))  # left:0, right: others

        all_parts_name2index = pickle.load(open('./{}_all_parts.name2index'.format(data_name), 'rb'))
        part_number = len(all_parts_name2index.keys())
        print(part_number)
        for mode in ['mine']:

            for part_name in range(1, part_number):
                embedding_1_name = '{}_{}'.format(data_name, 0)
                embedding_2_name = '{}_{}'.format(data_name, part_name)

                embedding_1 = torch.load(
                    './' + embedding_1_name + '_before_' + mode + '.embeddings')  # .cpu()  # others
                embedding_2 = torch.load('./' + embedding_2_name + '_before_' + mode + '.embeddings')  # .cpu()  # 0

                model = Matching(embedding_1.shape[1]).to(device)

                optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)

                model.train()
                for epoch in range(epoches):
                    optimizer.zero_grad()

                    train_loss = model(embedding_1, embedding_2, anchors_p)
                    print("{}|{}| part: {}/{}->0 | {}/{} | "
                          "train_loss: {:.8f}".format(mode, data_name, part_name, part_number, epoch, epoches,
                                                      train_loss.item()))

                    train_loss.backward()
                    optimizer.step()

                model.save_embeddings(mode=mode, embedding_1_name=embedding_1_name, embedding_2_name=embedding_2_name)
            # all_parts_name2index
            # {
            #     'part1': {
            #         1: 0,
            #         2: 1
            #     },
            #     'part2': {
            #         3: 0,
            #         4: 1
            #     }
            # }
            print('embedding合并...')

            embedding_list = []  #
            for part_name in range(part_number):
                emb = torch.load('./{}_{}_after_{}.embeddings'.format(data_name, part_name, mode))
                if part_name > 0:
                    emb = emb[shared_number:, :]

                embedding_list.append(emb)
            embedding_global = torch.cat(embedding_list, dim=0)
            torch.save(embedding_global, './{}_global_after_{}.embeddings'.format(data_name, mode))
