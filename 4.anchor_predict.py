import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn import metrics
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


class LinearClassifier(nn.Module):
    def __init__(self, d_embedding_after=100, d_hid=100, nclass=2, dropout=0.2):
        super(LinearClassifier, self).__init__()
        self.fc1 = nn.Linear(d_embedding_after * 2, d_hid)
        self.fc2 = nn.Linear(d_hid, nclass)
        self.dropout = dropout

    def forward(self, embedding_1, embedding_2, anchors_p_n):
        anchor_embedding = torch.cat([embedding_1[anchors_p_n[:, 0]],
                                      embedding_2[anchors_p_n[:, 1]]],
                                     dim=1)
        # print(anchor_embedding)
        anchor_embedding = self.fc1(anchor_embedding)
        anchor_embedding = F.relu(anchor_embedding)
        anchor_embedding = F.dropout(anchor_embedding, self.dropout)
        anchor_embedding = self.fc2(anchor_embedding)
        #anchor_embedding = F.dropout(anchor_embedding, self.dropout)  # 在ae上测试表明，把这两层去掉效果会更好
        #anchor_embedding = F.softmax(anchor_embedding,dim=1)
        # print(anchor_embedding)
        return anchor_embedding


if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    embedding_1_name = 'network_1'
    embedding_2_name = 'network_2'

    print("preparing link data...")
    # test_anchors_p_n = torch.load('./test_anchors_p_n.torch')
    # test_target = torch.load('./test_target.torch')
    test_anchors_p_n_df = pd.read_csv('./test_anchors_p_n.df', header=None)
    p_df = test_anchors_p_n_df[test_anchors_p_n_df[2] == 1]
    n_df = test_anchors_p_n_df[test_anchors_p_n_df[2] == 0]
    n_df = n_df.sample(frac=0.5, replace=True)
    test_anchors_p_n_df = pd.concat([p_df, n_df], axis=0)
    test_anchors_p_n_df = test_anchors_p_n_df.sample(frac=1.0)
    test_anchors_p_n = torch.from_numpy(np.array(test_anchors_p_n_df[[0, 1]]))
    test_target = torch.from_numpy(np.array(test_anchors_p_n_df[2])).view(-1).to(device)

    observed_anchors_p_n_df = pd.read_csv('./observed_anchors_p_n.df', header=None)
    p_df = observed_anchors_p_n_df[observed_anchors_p_n_df[2] == 1]
    n_df = observed_anchors_p_n_df[observed_anchors_p_n_df[2] == 0]
    n_df = n_df.sample(frac=0.5, replace=True)
    observed_anchors_p_n_df = pd.concat([p_df, n_df], axis=0)
    observed_anchors_p_n_df = observed_anchors_p_n_df.sample(frac=1.0)

    observed_anchors_p_n = torch.from_numpy(np.array(observed_anchors_p_n_df[[0, 1]]))
    observed_target = torch.from_numpy(np.array(observed_anchors_p_n_df[2])).view(-1).to(device)

    # observed_anchors_p_n = torch.load('./observed_anchors_p_n.torch').to(device)
    # observed_target = torch.load('./observed_target.torch').to(device)
    t_p = observed_target[observed_target == 1]
    t_n = observed_target[observed_target == 0]
    print('p/n:{}/{}'.format(t_p.shape[0], t_n.shape[0]))

    for mode in ['mah']:#,'gcn_only','ijcai16', 'hcn_only', 'mine']:

        embedding_1 = torch.load('./second_matching_network_1_after_{}.embeddings'.format(mode)).to(device)
        embedding_2 = torch.load('./second_matching_network_2_after_{}.embeddings'.format(mode)).to(device)
        print(embedding_1)
        print(embedding_2)
        model = LinearClassifier(d_embedding_after=embedding_1.shape[1], d_hid=100, nclass=2, dropout=0.00001).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.000005)

        model.train()
        best_f1 = 0.0
        best_pre = 0.0
        best_rec = 0.0
        epoches = 10000
        for epoch in range(epoches):
            optimizer.zero_grad()
            train_pre_score = model(embedding_1, embedding_2, observed_anchors_p_n)
            test_pre_score = model(embedding_1, embedding_2, test_anchors_p_n)
            train_loss = torch.nn.functional.cross_entropy(train_pre_score, observed_target)
            test_loss = torch.nn.functional.cross_entropy(test_pre_score, test_target)

            _, train_pre_class = train_pre_score.max(dim=1)
            _, test_pre_class = test_pre_score.max(dim=1)

            train_loss.backward()
            optimizer.step()

            train_maf1 = metrics.f1_score(observed_target.cpu(), train_pre_class.cpu(), average='macro')
            train_mapre = metrics.precision_score(observed_target.cpu(), train_pre_class.cpu(), average='macro')
            train_marec = metrics.recall_score(observed_target.cpu(), train_pre_class.cpu(), average='macro')
            # train_=metrics.auc()

            test_maf1 = metrics.f1_score(test_target.cpu(), test_pre_class.cpu(), average='macro')
            test_mapre = metrics.precision_score(test_target.cpu(), test_pre_class.cpu(), average='macro')
            test_marec = metrics.recall_score(test_target.cpu(), test_pre_class.cpu(), average='macro')

            if best_pre < test_mapre:
                best_pre = test_mapre
            if best_f1 < test_maf1:
                best_f1 = test_maf1
            if best_rec < test_marec:
                best_rec = test_marec
            if epoch % 1 == 0:
                print(mode + ": epoch:{}/{} "
                             "train_loss:{:.4f}|"
                             "test_loss:{:.4f} | "
                             "train_pre:{:.4f} | train_f1:{:.4f} | train_rec:{:.4f} | "
                             "test_pre:{:.4f} | test_f1:{:.4f} | test_rec:{:.4f} | "
                             "best_pre:{:.4f} | best_f1:{:.4f} | best_rec:{:.4f}".format(epoch, epoches,
                                                                                         train_loss.item(),
                                                                                         test_loss.item(),
                                                                                         train_mapre, train_maf1,
                                                                                         train_marec,
                                                                                         test_mapre, test_maf1,
                                                                                         test_marec,
                                                                                         best_pre, best_f1, best_rec))

        print(mode + ": best_pre:{:.4f} | best_f1:{:.4f} | best_rec:{:.4f}".format(best_pre, best_f1, best_rec))
