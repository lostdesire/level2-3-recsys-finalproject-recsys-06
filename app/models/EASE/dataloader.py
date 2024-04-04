import os
import pandas as pd
import numpy as np
from scipy import sparse


class DataLoader(): # 존재하지도 않는 user가 sparse matrix 상에 포함됨
    '''
    Load Movielens dataset
    '''
    def __init__(self, args):

        self.path = os.path.join(args.pro_dir, args.data)
        self.pro_dir = args.pro_dir
        assert os.path.exists(self.path), "Preprocessed files do not exist. Run data.py"

        self.n_items = self.load_n_items()

    def load_data(self, datatype='train'):
        if datatype == 'train':
            return self._load_train_data(self.path)
        elif datatype == 'validation':
            return self._load_tr_te_data(datatype)
        elif datatype == 'test':
            return self._load_tr_te_data(datatype)
        elif datatype == 'inference':
            return self._load_inf_data(datatype)
        else:
            raise ValueError("datatype should be in [train, validation, test, inference]")

    def load_n_items(self):
        unique_iid = list()
        with open(os.path.join(self.pro_dir, 'unique_iid.txt'), 'r') as f:
            for line in f:
                unique_iid.append(line.strip())
        n_items = len(unique_iid)
        return n_items

    def _load_train_data(self, path):

        tp = pd.read_csv(path)
        n_users = tp['uid'].nunique() + 1

        rows, cols = tp['uid'], tp['iid']
        data = sparse.csr_matrix((np.ones_like(rows),
                                 (rows, cols)), dtype='float64',
                                 shape=(n_users, self.n_items)) # uid,iid 1 채우고 나머지는 모두 0을 채우는 sparse interaction matrix
        return data

    def _load_tr_te_data(self, datatype='test'):
        tr_path = os.path.join(self.pro_dir, '{}_tr.csv'.format(datatype))
        te_path = os.path.join(self.pro_dir, '{}_te.csv'.format(datatype))

        tp_tr = pd.read_csv(tr_path)
        tp_te = pd.read_csv(te_path)

        start_idx = min(tp_tr['uid'].min(), tp_te['uid'].min())
        end_idx = max(tp_tr['uid'].max(), tp_te['uid'].max())

        rows_tr, cols_tr = tp_tr['uid'] - start_idx, tp_tr['iid']
        rows_te, cols_te = tp_te['uid'] - start_idx, tp_te['iid']

        data_tr = sparse.csr_matrix((np.ones_like(rows_tr),
                                    (rows_tr, cols_tr)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        data_te = sparse.csr_matrix((np.ones_like(rows_te),
                                    (rows_te, cols_te)), dtype='float64', shape=(end_idx - start_idx + 1, self.n_items))
        
        return data_tr, data_te
    
    def _load_inf_data(self, datatype='inference'):
        inf_path = os.path.join(self.pro_dir, 'inference.csv')

        data_inf = pd.read_csv(inf_path)

        n_rows = data_inf.uid.nunique()
        n_cols = data_inf.iid.nunique()

        rows = data_inf['uid']
        cols = data_inf['iid']

        data_inf = sparse.csr_matrix((np.ones_like(rows),
                                    (rows, cols)), dtype='float64', shape=(n_rows, n_cols))

        return data_inf
    
    def data_loading(self):
        n_items = self.load_n_items()
        train_data = self.load_data('train')
        vad_data_tr, vad_data_te = self.load_data('validation')
        test_data_tr, test_data_te = self.load_data('test')
        data_inf = self.load_data('inference')

        return n_items, train_data, vad_data_tr, vad_data_te, test_data_tr, test_data_te, data_inf