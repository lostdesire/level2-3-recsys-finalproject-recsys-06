from torch import nn
import torch.nn.functional as F
import torch
import numpy as np
import scipy.sparse as sp


# closed form: fitting -> predict from input
class EASE(nn.Module):
    def __init__(self, n_users, n_items):
        super(EASE, self).__init__()

        self.iid_name = 'item'
        self.uid_name = 'user'

        self.user_num = n_users
        self.item_num = n_items

        self.reg_weight = 100 # 이거 계속 바꿔봐야 해!

        self.topk = 10

    # train
    def fit(self, X):
        '''
        fit closed form parameters
        X: sparse matrix (user_num * item_num)
        '''

        G = X.T @ X # item_num * item_num
        G += self.reg_weight * sp.identity(G.shape[0])
        G = G.todense()

        P = np.linalg.inv(G)
        B = -P / np.diag(P) # equation 8 in paper: B_{ij}=0 if i = j else -\frac{P_{ij}}{P_{jj}}
        np.fill_diagonal(B, 0.)

        self.item_similarity = B # item_num * item_num
        self.item_similarity = np.array(self.item_similarity)
        self.interaction_matrix = X # user_num * item_num

    # 유저 u와 아이템 i 사이의 score를 예측
    def predict(self, u, i):
        return self.interaction_matrix[u, :].multiply(self.item_similarity[:, i].T).sum(axis=1).getA1()[0]

    # 유저(us)들의 후보군(cands_ids)들에 대한 ranking
    def rank(self, test_loader):
        rec_ids = None

        for us, cands_ids in test_loader:
            us = us.numpy()
            cands_ids = cands_ids.numpy()

            slims = np.expand_dims(self.interaction_matrix[us, :].todense(), axis=1) # batch * item_num -> batch * 1* item_num
            sims = self.item_similarity[cands_ids, :].transpose(0, 2, 1) # batch * cand_num * item_num -> batch * item_num * cand_num
            scores = np.einsum('BNi,BiM -> BNM', slims, sims).squeeze(axis=1) # batch * 1 * cand_num -> batch * cand_num
            rank_ids = np.argsort(-scores)[:, :self.topk]
            rank_list = cands_ids[np.repeat(np.arange(len(rank_ids)).reshape(-1, 1), rank_ids.shape[1], axis=1), rank_ids]

            rec_ids = rank_list if rec_ids is None else np.vstack([rec_ids, rank_list])

        return rec_ids

    # trainset 유저 중 일부를 score + ranking
    def full_rank(self, u):
        scores = self.interaction_matrix[u, :] @ self.item_similarity
        scores[self.interaction_matrix.nonzero()] = -np.inf
        return np.argsort(-scores)[:, :self.topk]
    
    # 새로운 유저 score: validation
    def predict_new(self, new_interaction):
        scores = new_interaction @ self.item_similarity
        return scores  #[n_users, n_items]
    
    # 새로운 유저 score+ranking: inference
    def rank_new(self, new_interaction):
        scores = new_interaction @ self.item_similarity
        scores[new_interaction.nonzero()] = -np.inf
        return np.argsort(-scores)[:,:self.topk]
    
    # score + ranking
    def rank_all(self, data):
        scores = data @ self.item_similarity
        scores[data.nonzero()] = -np.inf
        return np.argsort(-scores)[:,:self.topk]