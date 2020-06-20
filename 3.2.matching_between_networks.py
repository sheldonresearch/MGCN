import torch
import torch.nn.functional as F
import torch.nn as nn
import pandas as pd
import numpy as np
import os
import pickle

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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

        #dots_p = torch.sum(torch.mul(x_1_p, x_2_p), dim=1)
        #positive_loss = torch.mean(-1.0 * F.logsigmoid(dots_p))
        #return positive_loss
        dis_p = torch.mean(F.pairwise_distance(x_1_p, x_2_p))
        return dis_p

    def save_embeddings(self, mode='gcn_only', embedding_1_name='network_1', embedding_2_name='network_2'):
        torch.save(self.embedding_1_after, './' + embedding_1_name + '_after_' + mode + '.embeddings')
        torch.save(self.embedding_2_after, './' + embedding_2_name + '_after_' + mode + '.embeddings')


if __name__ == "__main__":

    epoches_2=10000

    print('网络间matching...')

    observed_anchors_pd = pd.read_csv('./observed_anchors.positive', header=None)  # index form, [i,i 1]
    test_anchors_pd = pd.read_csv('./test_anchors.positive', header=None)
    observed_anchors_p = torch.from_numpy(np.array(observed_anchors_pd[[0, 1]]))
    test_anchors_p = torch.from_numpy(np.array(test_anchors_pd[[0, 1]]))

    for mode in ['mine']:

        embedding_1 = torch.load('./network_1_global_after_{}.embeddings'.format(mode))  # .cpu() # others
        embedding_2 = torch.load('./network_2_global_after_{}.embeddings'.format(mode))  # .cpu() # 0

        model = Matching(embedding_1.shape[1]).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)

        model.train()
        for epoch in range(epoches_2):
            optimizer.zero_grad()
            train_loss = model(embedding_1, embedding_2, observed_anchors_p)
            test_loss = model(embedding_1, embedding_2, test_anchors_p)

            print(mode + "{}/{} | "
                         "train_loss: {:.8f} | "
                         "test_loss: {:.8f}".format(epoch, epoches_2,
                                                    train_loss.item(),
                                                    test_loss.item()))

            train_loss.backward()
            optimizer.step()

        model.save_embeddings(mode=mode, embedding_1_name='second_matching_network_1',
                              embedding_2_name='second_matching_network_2')

        # './second_matching_network_1_after_gcn_only.embeddings'
